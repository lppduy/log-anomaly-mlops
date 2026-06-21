# Part 8: CI/CD + Monitoring

## Mục tiêu

Đóng vòng MLOps:
- **CI**: tự test khi push code
- **Docker**: deploy API reproducible
- **Monitoring**: theo dõi runtime vs train

## Kiến thức nền

### Pipeline hoàn chỉnh

```
Code push → GitHub Actions (CI)
Train     → MLflow (Part 6)
Deploy    → Docker API (Part 7)
Monitor   → /metrics + drift check (Part 8)
Retrain   → khi drift vượt ngưỡng (manual/CI trigger)
```

### Drift là gì?

Metric runtime **lệch nhiều** so với lúc train:
- `flag_rate` tăng đột ngột → model quá nhạy hoặc data thay đổi
- `avg_latency_ms` cao → API chậm, cần scale

## Cấu trúc Part 8

```
.github/workflows/ci.yml   # GitHub Actions
Dockerfile                 # Container API
docker-compose.yml         # Chạy local với volume mlruns + models
src/monitoring/
├── metrics.py             # in-memory counters
└── checks.py              # so sánh runtime vs train
scripts/explore_part8.py
```

## Chạy thử

```bash
# Monitoring (TestClient, không cần server)
python scripts/explore_part8.py

# Docker (cần mlruns + models local trước)
docker compose up --build

# CI chạy tự động khi push lên GitHub
```

## Endpoint mới

`GET /metrics`:

```json
{
  "total_requests": 3,
  "predict_requests": 3,
  "anomaly_flags": 1,
  "flag_rate": 0.3333,
  "avg_latency_ms": 12.5,
  "uptime_sec": 45.2
}
```

## Config monitoring

```yaml
monitoring:
  max_flag_rate_delta: 0.15   # cho phép lệch 15% so với train
  max_avg_latency_ms: 100     # SLA latency
```

## 3 câu tự kiểm tra

1. CI Part 8 test gì? (compile + Part 1 smoke)
2. Docker mount `mlruns/` để làm gì?
3. Khi nào cần retrain?

## Project hoàn thành

Sau Part 8 bạn đã đi full loop MLOps học tập:
**data → features → train → MLflow → API → monitor → (retrain)**
