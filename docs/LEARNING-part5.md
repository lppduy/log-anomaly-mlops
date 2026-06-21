# Part 5: Train ML Model

## Mục tiêu

Train **Isolation Forest** trên `X_train`, chấm anomaly trên `X_test`, so sánh với baseline.

## Kiến thức nền

### Isolation Forest là gì?

**Unsupervised** = không cần label.

Ý tưởng: log bất thường **dễ tách riêng** hơn log bình thường.

```
Random split data
  → log bình thường: cần nhiều lần split mới tách được
  → log bất thường:   ít split là tách được → score thấp (raw)
```

### Input / Output Part 5

```
Input:  X_train (1600, 63)  từ Part 4
Output: anomaly_score 0-1 cho mỗi log test
        + models/anomaly_model.joblib
```

### sklearn score_samples

```
score_samples càng THẤP → càng bất thường
```

Project đổi sang 0-1, **cao = bất thường** (giống baseline).

## Cấu trúc Part 5

```
src/models/
├── train.py       # build, fit, score
└── evaluate.py    # precision@k, flag rate

scripts/
├── run_train_part5.py
└── explore_part5.py
```

## Chạy thử

```bash
python scripts/run_features_part4.py   # nếu chưa có X
python scripts/run_train_part5.py
python scripts/explore_part5.py
```

## Đánh giá (quan trọng)

HDFS_2k **không có label anomaly thật**.

Part 5 dùng **proxy**: `is_rare=1` từ feature matrix.

```
Precision@100 = trong top 100 ML score cao nhất,
                bao nhiêu % có is_rare=1
```

Đây là proxy để so sánh ML vs baseline, **không phải metric production**.

## 3 câu tự kiểm tra

1. Isolation Forest cần label không?
2. ML input là gì? (text hay số?)
3. Tại sao so sánh với baseline v1/v2?

## Giải thích sâu

Xem [isolation-forest-deep-dive.md](./isolation-forest-deep-dive.md):
- Isolation Forest hoạt động thế nào (cây, path length)
- contamination là gì
- precision@K là gì
- model.joblib bản chất chứa gì

## Part 6 (tiếp theo)

- MLflow log params, metrics, model artifact
- Model registry Staging → Production
