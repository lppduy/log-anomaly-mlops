# Part 4: Feature Engineering

## Mục tiêu

Biến parsed logs (text) thành **ma trận số X** để ML model đọc được.

## Kiến thức nền

### ML không đọc text trực tiếp

```
Log: "Connection refused to redis:6379"
        ↓ feature engineering
X: [2, 1, 35, 0, 0, 1, 0, 1, 0, 0.12, 0.45, ...]
```

### Fit vs Transform (quan trọng!)

```
fit(train_logs)     → học template_count, TF-IDF vocab, service list
transform(any_logs) → áp dụng rules đã học
```

**Train set** học statistics. **Test set** chỉ transform, không fit lại.
→ Tránh data leakage (Part 3 baseline tính count trên cùng file = leakage).

## Features trong project

| Feature | Kiểu | Ý nghĩa |
|---------|------|---------|
| `template_count` | số | Template xuất hiện bao nhiêu lần trong train |
| `is_rare` | 0/1 | count <= threshold |
| `message_len` | số | Độ dài message |
| `level_*` | one-hot | INFO, WARN, ERROR |
| `service_*` | one-hot | Component/service |
| `tfidf_*` | số | Pattern text của template |

## Cấu trúc Part 4

```
src/features/
├── build_features.py   # LogFeatureBuilder (fit/transform)
└── split.py            # train/test split

scripts/
├── run_features_part4.py
└── explore_part4.py

Output:
├── models/feature_builder.joblib   # save builder (gitignored)
├── data/processed/features_hdfs.npz
└── data/processed/features_meta.json
```

## Chạy thử

```bash
python scripts/run_parse_part2.py      # nếu chưa có parsed
python scripts/run_features_part4.py
python scripts/explore_part4.py
```

## Đọc code theo thứ tự

1. `src/features/build_features.py` - LogFeatureBuilder
2. `src/features/split.py` - train/test split
3. `scripts/run_features_part4.py` - pipeline
4. `data/processed/features_meta.json` - xem shape

## 3 câu tự kiểm tra

1. Tại sao phải `fit` trên train, `transform` trên test?
2. `is_rare` khác gì baseline v2?
3. `X_train.shape` là gì? (samples x features)

## Bài tập nhỏ

1. Đổi `tfidf_max_features` 50 -> 10, chạy lại, so sánh `n_features`
2. Mở `features_meta.json`, đếm bao nhiêu `tfidf_` features
3. Trong explore output, tìm log `is_rare=1`, xem `template_count`

## Part 5 (tiếp theo)

Xem [LEARNING-part5.md](./LEARNING-part5.md)
