# Log Anomaly MLOps

Học MLOps end-to-end qua bài toán phát hiện log bất thường.

## Problem

Detect anomalous log lines in system logs.

## Input

Single log line:

```json
{
  "timestamp": "2026-06-20T10:00:05Z",
  "level": "ERROR",
  "service": "db",
  "message": "Connection refused to postgres:5432"
}
```

## Output

```json
{
  "anomaly_score": 0.92,
  "is_anomaly": true,
  "model_version": "baseline-v1"
}
```

## Success criteria (v1)

- Precision@100 >= 0.7 on HDFS test set
- API latency p95 < 100ms

## Non-goals (v1)

- Real-time streaming (Kafka)
- Root cause analysis
- LLM-based detection

## Quick start (Part 1)

```bash
cd zzz-code/log-anomaly-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/explore_part1.py
python scripts/run_baseline.py
```

## Quick start (Part 2)

```bash
python scripts/download_hdfs.py
python scripts/run_parse_part2.py
python scripts/explore_part2.py
```

## Quick start (Part 3)

```bash
python scripts/explore_part3.py
python scripts/run_baseline_v2.py
```

## Quick start (Part 4)

```bash
python scripts/run_features_part4.py
python scripts/explore_part4.py
```

## Quick start (Part 5)

```bash
python scripts/run_train_part5.py
python scripts/explore_part5.py
```

## Quick start (Part 6)

```bash
python scripts/run_mlflow_part6.py
python scripts/explore_part6.py

# UI (terminal khác, Python 3.13)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Quick start (Part 7)

```bash
python scripts/explore_part7.py
python scripts/run_api_part7.py

# Terminal khác (server đang chạy)
bash scripts/test_api_part7.sh
```

## Quick start (Part 8)

```bash
python scripts/explore_part8.py

# Docker (cần mlruns + models local)
docker compose up --build
```

Chi tiết học: [docs/](./docs/)

- [Part 1](./docs/LEARNING.md) | [Part 2](./docs/LEARNING-part2.md) | [Part 3](./docs/LEARNING-part3.md) | [Part 4](./docs/LEARNING-part4.md) | [Part 5](./docs/LEARNING-part5.md) | [Part 6](./docs/LEARNING-part6.md) | [Part 7](./docs/LEARNING-part7.md) | [Part 8](./docs/LEARNING-part8.md)
- [Giải thích sâu Isolation Forest](./docs/isolation-forest-deep-dive.md)

## Roadmap

| Part | Nội dung | Status |
|------|----------|--------|
| 1 | Framing + baseline | Done |
| 2 | Ingest + normalize + template | Done |
| 3 | EDA template + baseline v2 | Done |
| 4 | Feature engineering | Done |
| 5 | Train model | Done |
| 6 | MLflow | Done |
| 7 | FastAPI serving | Done |
| 8 | CI/CD + monitoring | Done |
