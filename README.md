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

Chi tiết học:
- [LEARNING.md](./LEARNING.md) (Part 1)
- [LEARNING-part2.md](./LEARNING-part2.md) (Part 2)
- [LEARNING-part3.md](./LEARNING-part3.md) (Part 3)

## Roadmap

| Part | Nội dung | Status |
|------|----------|--------|
| 1 | Framing + baseline | Done |
| 2 | Ingest + normalize + template | Done |
| 3 | EDA template + baseline v2 | Done |
| 4 | Feature engineering | Todo |
| 5 | Train model | Todo |
| 6 | MLflow | Todo |
| 7 | FastAPI serving | Todo |
| 8 | CI/CD + monitoring | Todo |
