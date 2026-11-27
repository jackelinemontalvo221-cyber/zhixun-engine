# 成员 1 参考实现与取舍

| 项目 | 借鉴点 | 是否引入依赖 |
| --- | --- | --- |
| Temporal | workflow/activity 分离、每步 journal、稳定幂等键、重试分类、信号 | 否，只借语义 |
| DVC | 内容寻址、stage 缓存、哈希指针 | 否 |
| MLflow Model Registry | alias-first、`previous` 指针、确定性回滚 | 否 |
| Pandera / Great Expectations | 声明式校验、一次报所有错、fail-closed | 否，手写校验器 |
| Kedro / Prefect | 数据目录抽象、任务状态、retry policy | 否 |
| `python-statemachine` | 显式状态转移表 | 否 |
| `eventsourcing` | append-only 事件日志、乐观并发 | 否 |
| `filelock` | 跨平台文件锁 | 否，POSIX 用 `fcntl` |
| `tenacity` | 退避重试 | 否，编排器只保留配置项 |
| `structlog` | 结构化 JSON 日志 | 否，标准库 `logging` + `events.jsonl` |
| Hypothesis | 属性测试 | 否（当前用固定夹具） |
| distroless / hadolint / dive | 最小离线镜像、Dockerfile 检查、体积分析 | 交付阶段 |

原则：正式镜像只依赖 Python 标准库 + NumPy；其余只借设计，不引入运行时依赖。
