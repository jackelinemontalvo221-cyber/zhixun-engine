"""Local two-stage state machine and CLI.

The official platform drives rounds through ``round_runner`` instead; this module
keeps the same pipeline runnable offline with a local manifest source.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import uuid
import hashlib
from dataclasses import replace
from pathlib import Path

from .budget import BudgetState
from .codec import bind_stage, merge_overlay, normalize_stage, profile_from_raw
from .config import HarnessConfig, load_config, load_overlay
from .contracts import (
    ActiveModel,
    StageBundle,
    StageProfile,
    StageRecord,
    now_utc,
)
from .errors import (
    BudgetExceeded,
    ConfigError,
    FatalContractError,
    HarnessError,
    StageDataError,
)
from .fakes import FakeEvaluator, FakeModelGateway, FakePipeline
from .sources import LocalManifestSource
from .store import ArtifactStore
from .components import build_pipeline, build_evaluator
from .round_state import RoundState, RoundContext, track_rows

LOGGER = logging.getLogger("zhixun_harness")

_WAITING = "WAITING_STAGE"
_VALIDATED = "VALIDATED"
_COMMITTED = "COMMITTED"
_FINISHED = "FINISHED"
_FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
_FAILED_FATAL = "FAILED_FATAL"

_STOP = False


def _handle_signal(_signum, _frame):
    global _STOP
    _STOP = True


def _apply_overlay(profile: StageProfile, overlay: dict | None) -> StageProfile:
    if not overlay or overlay.get("task_id") != profile.task_id:
        return profile
    data = {key: value for key, value in merge_overlay(profile.to_dict(), overlay).items()
            if key in StageProfile.__dataclass_fields__}
    return StageProfile.from_dict(data)


def _active_initial(task_id: str, checkpoint_path: str | Path) -> ActiveModel:
    from .hashing import file_sha256
    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"initial checkpoint is not a regular file: {path}")
    return ActiveModel(
        task_id=task_id,
        checkpoint_uri=str(path),
        checkpoint_sha256=file_sha256(path),
        stage_id="initial",
        activated_at=now_utc(),
        reason="initial",
    )


def run(
    source,
    store: ArtifactStore,
    pipeline,
    evaluator,
    config: HarnessConfig,
    *,
    max_stages: int | None = None,
    initial_checkpoint: str | Path | None = None,
    overlay: dict | None = None,
) -> int:
    global _STOP
    _STOP = False
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    budget = BudgetState.from_config(config.budget)
    history: list[StageRecord] = []
    active = store.active_model()
    if active is None:
        candidate_path = initial_checkpoint or config.initial_checkpoint
        if candidate_path is None:
            raise ConfigError("initial_checkpoint is required for a new artifact root")
        active = store.ensure_initial(candidate_path)
    limiter = max_stages if max_stages is not None else config.budget.get("max_stages")

    while not _STOP:
        descriptor = source.next()
        if descriptor is None:
            break
        if limiter is not None and budget.stages >= limiter:
            break
        budget.begin_stage()
        txn = None
        try:
            raw = source.read(descriptor)
            profile = _apply_overlay(profile_from_raw(raw), overlay)
            bundle = bind_stage(normalize_stage(raw, profile), descriptor.stage_id)
            if store.has_committed(bundle):
                source.ack(descriptor.stage_id, "reused")
                LOGGER.info("stage %s reused", descriptor.stage_id)
                continue

            txn = store.begin_stage(descriptor, bundle)
            txn.append_event({"event": "bundle_ready", "state": _VALIDATED,
                              "data": {"sample_count": bundle.sample_count,
                                       "canonical_sha256": bundle.canonical_sha256}})
            started = time.monotonic()
            context = None
            if hasattr(pipeline, "bind_round"):
                state = RoundState(store.root / "member-state" / hashlib.sha256(bundle.task_id.encode()).hexdigest()[:20])
                validation = None
                if bundle.eval_features is not None and len(bundle.eval_ids):
                    validation = replace(bundle, stage_id=bundle.stage_id + "-validation",
                        ids=bundle.eval_ids, features=bundle.eval_features, targets=bundle.eval_targets,
                        group_ids=bundle.eval_group_ids, sample_count=len(bundle.eval_ids))
                deltas = track_rows(state, {"train": bundle, "validation": validation})
                ref = state.read("reference.json", {})
                if ref.get("checkpoint_sha256") != active.checkpoint_sha256:
                    ref = {}
                gate = dict(config.gate, metric_name="macro_f1" if bundle.task_type == "classification" else "rmse",
                            direction="maximize" if bundle.task_type == "classification" else "minimize")
                context = RoundContext(bundle.task_id, budget.stages, state, active,
                    checkpoint_path=store.root / active.checkpoint_uri, eval_bundle=validation,
                    history=[dict(r.to_dict(), id=r.record_id) for r in history], reference=ref,
                    new_rows=deltas["train"], new_validation_rows=deltas.get("validation"),
                    gateway=getattr(pipeline, "gateway", None), gate=gate)
                pipeline.bind_round(context)
                if hasattr(evaluator, "bind_round"):
                    evaluator.bind_round(context)
            current = evaluator.evaluate(active, context.eval_bundle) if context and context.eval_bundle is not None else (
                {} if context else evaluator.evaluate(active, bundle))
            diagnosis = pipeline.diagnose(bundle, current, budget)
            txn.write_json("diagnosis.json", diagnosis.to_dict())
            decision = pipeline.decide(diagnosis, history, budget)
            txn.write_json("decision.json", decision.to_dict())
            active_before = active.checkpoint_sha256

            candidate = None
            evaluation = None
            final_action = decision.action if decision.action == "no_update" else "failed"
            active_after = active.checkpoint_sha256
            pending = None
            if decision.action == "no_update":
                final_action = "no_update"
            elif decision.action == "rollback":
                if active.previous is None:
                    raise FatalContractError("no previous active model to roll back to")
                active_after = active.previous["checkpoint_sha256"]
                pending = "rollback"
                final_action = "rollback"
            else:
                budget.begin_candidate()
                candidate = pipeline.train_candidate(bundle, decision, active, budget)
                if candidate is None:
                    raise StageDataError("pipeline returned no candidate")
                txn.write_candidate(candidate)
                evaluation = pipeline.evaluate_and_commit(bundle, candidate, active,
                    context.gate if context else config.gate, **({"eval_bundle": context.eval_bundle} if context else {}))
                txn.write_json("evaluation.json", evaluation.to_dict())
                if evaluation.gate == "promote":
                    active_after = candidate.checkpoint_sha256
                    pending = ("activate", candidate)
                    final_action = "promote"
                elif evaluation.gate == "rollback":
                    if active.previous is None:
                        raise FatalContractError("no previous active model to roll back to")
                    active_after = active.previous["checkpoint_sha256"]
                    pending = "rollback"
                    final_action = "rollback"
                else:
                    final_action = "keep"

            record = StageRecord(
                record_id=f"rec-{uuid.uuid4()}",
                run_id=store.run_id,
                task_id=descriptor.task_id,
                stage_id=descriptor.stage_id,
                attempt_id="a1",
                bundle_sha256=bundle.canonical_sha256,
                diagnosis_id=diagnosis.diagnosis_id,
                decision_id=decision.decision_id,
                final_action=final_action,
                active_before_sha256=active_before,
                active_after_sha256=active_after,
                timings={"total": time.monotonic() - started},
                events_count=0,
                candidate_id=candidate.candidate_id if candidate else None,
                evaluation_id=evaluation.evaluation_id if evaluation else None,
            )
            txn.write_json("record.json", record.to_dict())
            txn.append_event({"event": "stage_committed", "state": _COMMITTED,
                              "data": {"final_action": final_action}})
            txn.commit()
            txn = None
            if isinstance(pending, tuple):
                active = store.activate(pending[1], descriptor.stage_id)
            elif pending == "rollback":
                active = store.rollback()
            history.append(record)
            if context is not None:
                ref = dict(context.reference)
                if final_action == "promote":
                    ref = {"checkpoint_sha256": active.checkpoint_sha256,
                           "at_activation": evaluation.candidate_metrics}
                else:
                    ref.update(checkpoint_sha256=active.checkpoint_sha256, last_round=current)
                context.state.write("reference.json", ref)
            source.ack(descriptor.stage_id, "done")
            LOGGER.info("stage %s committed: %s", descriptor.stage_id, final_action)
        except BudgetExceeded:
            if txn is not None:
                txn.abort()
            raise
        except FatalContractError:
            if txn is not None:
                txn.abort()
            raise
        except (StageDataError, HarnessError) as exc:
            if txn is not None:
                txn.abort()
            LOGGER.error("stage %s failed: %s", descriptor.stage_id, exc)
            source.ack(descriptor.stage_id, "failed")
            if config.retry.get("on_stage_error", "continue") == "stop":
                return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m zhixun_harness.orchestrator")
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--dev-profile")
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        overlay = load_overlay(args.dev_profile) if args.dev_profile else None
        initial = args.initial_checkpoint or config.initial_checkpoint
        if initial is None and overlay and overlay.get("initial_checkpoint"):
            initial = Path(overlay.get("package_dir", ".")) / overlay["initial_checkpoint"]
        source = LocalManifestSource(args.stage_root)
        with ArtifactStore(args.artifact_root, args.run_id) as store:
            pipeline = build_pipeline(config, FakeModelGateway())
            run(source, store, pipeline, build_evaluator(config), config,
                max_stages=args.max_stages, initial_checkpoint=initial, overlay=overlay)
        return 0
    except BudgetExceeded as exc:
        LOGGER.error("%s", exc)
        return 3
    except (ConfigError, FatalContractError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except HarnessError as exc:
        LOGGER.error("%s", exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
