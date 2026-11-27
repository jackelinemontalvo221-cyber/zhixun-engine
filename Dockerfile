# Harness image for contest.harness.workspace.v2.
# Build from the repository root (records the commit in the image and the evidence):
#   bash scripts/build_image.sh            # = docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) ...
# Then check it before uploading:
#   bash scripts/verify_image.sh zhixun-harness:0.1.0
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Override for restricted networks, e.g.
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG GIT_COMMIT=unknown
# Platform trial images: --build-arg HARNESS_CONFIG=/app/configs/harness.trial.json
ARG HARNESS_CONFIG=/app/configs/harness.full.json

LABEL org.opencontainers.image.revision=$GIT_COMMIT

WORKDIR /opt/zhixun

COPY contest-harness-sdk-1.0.4.tar.gz /tmp/contest-harness-sdk-1.0.4.tar.gz
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" /tmp/contest-harness-sdk-1.0.4.tar.gz

COPY pyproject.toml /opt/zhixun/pyproject.toml
COPY contest_submission.py /opt/zhixun/contest_submission.py
COPY zhixun_harness /opt/zhixun/zhixun_harness
COPY member2 /opt/zhixun/member2
COPY member3 /opt/zhixun/member3
COPY member4 /opt/zhixun/member4
COPY member5 /opt/zhixun/member5
COPY vendor /opt/zhixun/vendor
COPY configs /app/configs
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" . \
 && python3 -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], 'zhixun_sigterm.pth').write_text('import zhixun_harness.sigterm\\n')"
# ^ In a container the harness is PID 1, which ignores SIGTERM unless a handler is
#   installed, and the SDK imports our module only after the start frame.  The .pth
#   hook installs the handler as soon as Python starts, so a platform stop is prompt.

COPY app/harness.py /app/harness.py
COPY contest_submission.py /app/contest_submission.py

RUN mkdir -p /workspace && chown 65534:65534 /workspace

ENV CONTEST_SUBMISSION=contest_submission \
    ZHIXUN_BASELINES=/opt/zhixun/vendor \
    CONTEST_CONFIG=$HARNESS_CONFIG \
    ZHIXUN_GIT_COMMIT=$GIT_COMMIT
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
