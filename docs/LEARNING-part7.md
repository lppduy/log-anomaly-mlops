# Part 7: FastAPI Serving

## Mục tiêu

Deploy model qua HTTP API:
- Load model + feature_builder từ **MLflow Staging**
- `POST /predict` nhận 1 log line → trả `anomaly_score`

## Kiến thức nền

### Pipeline inference

```
POST /predict {message, level, ...}
  → Drain3 parse template (state học từ HDFS train)
  → feature_builder.transform() → 63 số
  → Isolation Forest score
  → normalize bằng bounds từ train
  → {anomaly_score, is_anomaly, model_version}
```

`models/drain3_hdfs.bin` lưu state Drain3 từ train, API load cùng lúc với model.
Client vẫn có thể gửi sẵn `template` để skip parse.

### Tại sao cần score bounds?

Part 5 normalize score theo min/max của **cả batch test**.
Serve 1 log không có batch → dùng `score_inverted_min/max` học từ **X_train**.

### Part 6 vs Part 7

| | Part 6 | Part 7 |
|---|--------|--------|
| Output | MLflow registry | HTTP API |
| User | Data scientist | Oncall / service khác |
| Load model | Manual / UI | Auto lúc startup |

## Cấu trúc Part 7

```
src/serving/
├── app.py              # FastAPI routes
├── load_artifacts.py   # load MLflow Staging hoặc local
├── predictor.py        # inference 1 log
└── schemas.py          # request/response

scripts/
├── run_api_part7.py    # uvicorn
├── explore_part7.py    # TestClient smoke test
└── test_api_part7.sh   # curl khi server chạy
```

## Chạy thử

```bash
python scripts/run_mlflow_part6.py   # nếu chưa có registry

# Smoke test (không cần mở server)
python scripts/explore_part7.py

# Server thật
python scripts/run_api_part7.py
# Docs: http://127.0.0.1:8000/docs

# Terminal khác
bash scripts/test_api_part7.sh
```

## Config

`configs/project.yaml`:

```yaml
serving:
  host: 127.0.0.1
  port: 8000
  source: mlflow      # hoặc local
  model_stage: Staging
```

## API

### POST /predict

Request:

```json
{
  "timestamp": "2026-06-21T10:00:00Z",
  "level": "INFO",
  "service": "dfs.DataNode",
  "message": "Receiving block blk_123 ..."
}
```

Response:

```json
{
  "anomaly_score": 0.42,
  "is_anomaly": false,
  "model_version": "log-anomaly-detector-v2-Staging",
  "template": "Receiving block <*> src: <*> dest: <*>",
  "raw_score": -0.31
}
```

## 3 câu tự kiểm tra

1. API load model từ đâu lúc startup?
2. Tại sao cần cả feature_builder lẫn isolation forest?
3. `source: local` vs `source: mlflow` khác gì?

## Part 8 (tiếp theo)

- CI/CD train + deploy
- Monitoring drift + retrain trigger
