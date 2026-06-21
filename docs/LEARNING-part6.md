# Part 6: MLflow Tracking + Model Registry

## Mục tiêu

Sau Part 5, model nằm local `models/anomaly_model.joblib`. Part 6 trả lời:

1. **Lần train này metric bao nhiêu?** (so với lần trước)
2. **Model version nào đang Staging?** (cho API Part 7)

## Kiến thức nền

### Part 5 vs Part 6

| | Part 5 | Part 6 |
|---|--------|--------|
| Output | file local `.joblib` | + MLflow run + registry |
| So sánh run | Khó (tự ghi tay) | UI MLflow |
| Deploy | path cứng | load từ registry |

### MLflow gồm 3 phần

```
Experiment   = nhóm các lần train (vd: log-anomaly-mlops)
Run          = 1 lần train cụ thể (params + metrics + artifacts)
Model Registry = quản lý version Staging / Production
```

### Artifact log gì?

```
run/
├── isolation_forest/     # sklearn model (Part 7 load cái này)
├── feature_builder/      # joblib Part 4 (cần kèm khi serve)
└── reports/
    ├── train_metrics.json
    ├── predictions_test.jsonl
    └── _mlflow_prediction_sample.jsonl
```

## Cấu trúc Part 6

```
src/
├── models/experiment.py      # train pipeline dùng chung Part 5 + 6
└── tracking/
    └── mlflow_tracker.py     # log run, register, promote Staging

scripts/
├── run_mlflow_part6.py
└── explore_part6.py
```

## Chạy thử

```bash
python scripts/run_features_part4.py   # nếu chưa có X
python scripts/run_mlflow_part6.py
python scripts/explore_part6.py

# UI (terminal khác)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# mở http://127.0.0.1:5000
```

## Đọc code theo thứ tự

1. `src/models/experiment.py` - logic train (refactor từ Part 5)
2. `src/tracking/mlflow_tracker.py` - log params/metrics/artifacts
3. `scripts/run_mlflow_part6.py` - orchestration
4. `configs/project.yaml` section `mlflow:`

## Params vs Metrics log gì?

**Params** (cấu hình, không đổi sau train):
- `n_estimators`, `contamination`, `tfidf_max_features`...

**Metrics** (kết quả đo):
- `flag_rate_ml`, `precision_at_k_proxy_ml`, `overlap_with_v2`

## Model Registry flow

```
Train → log_model(registered_model_name=...)
      → version mới (vd: v1)
      → transition → Staging
Part 7 → load model Staging → serve API
```

Production promote thủ công hoặc qua CI (Part 8).

## 3 câu tự kiểm tra

1. Params khác metrics thế nào?
2. Tại sao log cả `feature_builder` artifact?
3. Part 7 load model từ đâu thay vì `models/anomaly_model.joblib`?

## Bài tập nhỏ

1. Đổi `contamination: 0.05 → 0.10` trong config, chạy lại Part 6, so 2 run trên UI
2. Trong UI, mở run mới → tab Artifacts → tải `reports/train_metrics.json`
3. Model Registry → xem version Staging trỏ run nào

## Part 7 (tiếp theo)

- FastAPI `/predict`
- Load model + feature_builder từ MLflow registry

Xem [LEARNING-part7.md](./LEARNING-part7.md)
