"""把评测平台上的运行报告（试运行 / 正式评测）上传到 SwanLab 公开项目。

赛题要求“全部训练、微调、调整与评估实验”在云端实验记录平台完整记录。评测容器离线运行，
不能也不应在容器里上报，所以每次平台运行结束后，在自己电脑上用这个脚本把平台报告补记到
SwanLab：每个“运行 × 数据集”生成一个实验，逐轮记录决策、门控、公开验证指标、候选指标、
资源用量和本轮假设/理由，最后记录隐藏测试集指标与总用量。

报告文件来自平台“评测运行 → 查看报告”背后的接口 /api/v1/runs/<运行号>/report，
保存为 platform-run-<运行号>.json。

用法（已激活装有 swanlab 的虚拟环境）：
    swanlab login                     # 第一次需要
    python scripts/upload_platform_runs.py 报告1.json 报告2.json ...
    python scripts/upload_platform_runs.py --dry-run 报告.json   # 只打印将上传的内容
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = "zhixun-engine"
MAX_TEXT = 3000


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _evidence(round_: dict) -> dict:
    configuration = round_.get("configuration") if isinstance(round_.get("configuration"), dict) else {}
    evidence = configuration.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def round_rows(dataset: dict) -> list[dict]:
    """One flat row of numbers per round, plus the texts kept for the record."""
    rows = []
    for round_ in sorted(dataset.get("rounds") or [], key=lambda item: item.get("round_index") or 0):
        evidence = _evidence(round_)
        index = int(round_.get("round_index") or len(rows) + 1)
        final = str(evidence.get("final_action") or "")
        action = str(evidence.get("action") or "")
        numbers = {
            "decision/trained": int(action not in ("", "no_update", "None")),
            "decision/promoted": int(final == "promote"),
            "decision/rolled_back": int(final == "rollback"),
            "decision/failed": int(final == "failed"),
            "decision/candidates": len(round_.get("candidates") or []),
            "decision/model_calls": len(round_.get("model_call_ids") or []),
            "decision/history_refs": len(round_.get("history_ids") or []),
        }
        for name, value in (round_.get("validation_metrics") or {}).items():
            if _num(value) is not None:
                numbers[f"selected_validation/{name}"] = value
        for name, value in (evidence.get("parent_metrics") or {}).items():
            if _num(value) is not None and ":" not in name:
                numbers[f"parent_validation/{name}"] = value
        for name, value in (evidence.get("candidate_metrics") or {}).items():
            if _num(value) is not None and ":" not in name:
                numbers[f"candidate_validation/{name}"] = value
        if _num(evidence.get("delta")) is not None:
            numbers["gate/delta"] = evidence["delta"]
        for name, value in (evidence.get("budget_used") or {}).items():
            if _num(value) is not None:
                numbers[f"budget_used/{name}"] = value
        text = (
            f"第 {index} 轮 | 动作 {action or '-'} | 结果 {final or '-'} | 门控 {evidence.get('gate', '-')} | "
            f"漂移 {evidence.get('drift_type', '-')}/{evidence.get('severity', '-')} | "
            f"选中 {round_.get('selected_artifact_id', '-')}\n"
            f"假设：{round_.get('hypothesis', '')}\n理由：{round_.get('reason', '')}"
        )[:MAX_TEXT]
        rows.append({"round": index, "numbers": numbers, "text": text, "final_action": final, "action": action})
    return rows


def experiments(report: dict) -> list[dict]:
    run = report.get("run") or {}
    body = report.get("report") or {}
    workspace = body.get("workspace_report") or {}
    costs = body.get("costs") or {}
    out = []
    for dataset in workspace.get("datasets") or []:
        key = str(dataset.get("key"))
        rows = round_rows(dataset)
        summary = {f"hidden_test/{name}": value for name, value in (dataset.get("metrics") or {}).items()
                   if _num(value) is not None}
        summary.update({
            "resource/cpu_seconds": _num(dataset.get("cpu_seconds")),
            "resource/wall_seconds": _num(dataset.get("wall_seconds")),
            "resource/prediction_count": _num(dataset.get("prediction_count")),
            "record/complete": int(bool(dataset.get("record_complete"))),
            "summary/trained_rounds": sum(row["numbers"]["decision/trained"] for row in rows),
            "summary/promoted_rounds": sum(row["numbers"]["decision/promoted"] for row in rows),
            "summary/failed_rounds": sum(row["numbers"]["decision/failed"] for row in rows),
        })
        out.append({
            "name": f"PLAT-{str(run.get('mode', 'run'))}-v{run.get('submission_version', '?')}-{str(run.get('id', ''))[:8]}-{key}",
            "description": (f"评测平台{('试运行' if run.get('mode') == 'trial' else '正式评测')}，"
                            f"提交版本 v{run.get('submission_version')}，数据集 {key}，"
                            f"运行开始 {run.get('created_at')}；数据来自平台运行报告（逐轮实验记录与隐藏集指标）"),
            "config": {
                "experiment": "platform-run",
                "run_id": run.get("id"),
                "mode": run.get("mode"),
                "contest_stage": run.get("contest_stage"),
                "submission_version": run.get("submission_version"),
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "dataset": key,
                "model_identity": body.get("model_identity"),
                "primary_metric": report.get("metric"),
                "metric_direction": report.get("metric_direction"),
                "run_tokens": {name: costs.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")},
            },
            "tags": ["platform", str(run.get("mode")), f"task-{key}", f"v{run.get('submission_version')}"],
            "rows": rows,
            "summary": {name: value for name, value in summary.items() if value is not None},
        })
    return out


def upload(items: list[dict], dry_run: bool) -> None:
    if dry_run:
        for item in items:
            print("==", item["name"], "|", item["description"])
            for row in item["rows"]:
                print("  ", row["round"], row["action"], row["final_action"], len(row["numbers"]), "numbers")
            print("   summary:", item["summary"])
        return
    import swanlab

    for item in items:
        swanlab.init(mode="online", reinit=True, public=True, project=PROJECT, name=item["name"],
                     description=item["description"], config=item["config"], tags=item["tags"])
        for row in item["rows"]:
            swanlab.log(row["numbers"], step=row["round"])
            swanlab.log({"record/round": swanlab.Text(row["text"])}, step=row["round"])
        swanlab.log(item["summary"], step=100)
        swanlab.finish()
        print("已上传", item["name"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    items = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if "report" not in report or "run" not in report:
            print(f"{path}: 不是平台运行报告（缺少 run/report）", file=sys.stderr)
            return 2
        items.extend(experiments(report))
    upload(items, args.dry_run)
    print(f"共 {len(items)} 个实验（项目 {PROJECT}）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
