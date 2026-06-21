# Isolation Forest, contamination, precision@K: giải thích sâu

Tài liệu bổ sung cho [LEARNING-part5.md](./LEARNING-part5.md).

---

## 1. Isolation Forest hoạt động thế nào?

### Ý tưởng gốc (Isolation Forest, Liu et al.)

Anomaly = điểm **dễ bị cô lập** (isolate) khỏi phần còn lại.

Thay vì hỏi "giống cluster nào?" (như K-Means), IF hỏi: **"Cần bao nhiêu bước cắt ngẫu nhiên để tách riêng điểm này?"**

```
Ít bước cắt  →  dễ cô lập  →  bất thường
Nhiều bước   →  nằm giữa đám đông  →  bình thường
```

### 1 cây Isolation Tree làm gì?

Với mỗi node, chọn **ngẫu nhiên**:
1. Một **feature** (cột trong X, ví dụ `template_count`)
2. Một **ngưỡng split** nằm giữa min và max của feature đó

```
                    template_count <= 150?
                   /                        \
              YES /                          \ NO
                /                            \
    is_rare <= 0.5?                    tfidf_block <= 0.3?
         ...                                  ...
              \                              /
               → điểm bị tách ra ở lá (leaf)
```

**Path length** = số lần split từ gốc xuống lá để cô lập 1 điểm.

| Path length | Ý nghĩa |
|-------------|---------|
| Ngắn (2-3) | Điểm dễ tách → anomaly |
| Dài (8-10) | Điểm nằm giữa đám đông → normal |

### Tại sao cần nhiều cây (n_estimators=100)?

1 cây = 1 lần cắt ngẫu nhiên → nhiễu cao.

100 cây → lấy **trung bình path length** → ổn định hơn.

```python
n_estimators=100  # 100 cây, vote/trung bình path length
```

Trong project: `configs/project.yaml` → `model.n_estimators: 100`.

### score_samples trong sklearn

```
score ≈ -2^(-E[path_length] / c(n))
```

- `E[path_length]`: path length trung bình qua các cây
- Càng **thấp** → càng bất thường

Project đảo và normalize trong `src/models/train.py`:

```python
inverted = -raw                    # đảo chiều
anomaly_score = (inverted - min) / (max - min)   # scale 0-1
```

→ **Cao = bất thường**, thống nhất với baseline.

### Ví dụ với log HDFS (trực giác)

Log bình thường (train):
```
template_count=254, is_rare=0, level_INFO=1, tfidf giống đám đông
→ nằm giữa cloud → path dài → score thấp
```

Log bất thường (test):
```
template_count=0, is_rare=1, message_len=120, tfidf lạ
→ xa đám đông → path ngắn → score cao
```

ML có thể flag log mà **is_rare=0** nếu combination 63 features vẫn lạ (TF-IDF + message_len + service...).

---

## 2. contamination là gì?

### Định nghĩa

```python
IsolationForest(contamination=0.05)
```

**Giả định:** khoảng **5%** data train là outlier/anomaly.

### Dùng để làm gì?

1. **Calib threshold nội bộ** khi gọi `model.predict()` (trả -1/1)
2. **Bias tree building** trong một số implementation

sklearn dùng contamination để xác định ngưỡng offset cho `predict()`:
- Sort score_samples
- Lấy percentile tương ứng contamination làm cutoff

### Trong project

```yaml
# configs/project.yaml
model:
  contamination: 0.05
```

Chúng ta **không dùng trực tiếp** `model.predict()` làm output chính. Dùng `score_samples` + normalize + threshold 0.5 từ config `problem.anomaly_threshold`.

### Chọn contamination thế nào?

| Giá trị | Khi nào |
|---------|---------|
| 0.01 | Tin rằng anomaly rất hiếm (<1%) |
| 0.05 | Default hợp lý cho log monitoring |
| 0.10 | Nghi ngờ nhiều noise |
| "auto" | sklearn tự chọn (không khuyên khi học) |

**Quan trọng:** contamination **không phải label thật**. Chỉ là prior/giả định. Sai contamination → flag quá nhiều hoặc quá ít.

### Ví von

Phòng thi 100 học sinh, bạn **giả định** ~5 học sinh có bài làm bất thường. Không biết ai, nhưng model dùng giả định đó để calibrate ngưỡng.

---

## 3. precision@K là gì?

### Bối cảnh anomaly detection

Production thường không alert **tất cả** log score cao. Chỉ xem **top K** (K=100 log đáng nghi nhất).

### Công thức

```
Precision@K = (số log đúng anomaly trong top K) / K
```

Trong code (`src/models/evaluate.py`):

```python
top_idx = np.argsort(scores)[-k:][::-1]   # K score cao nhất
precision = y_true[top_idx].mean()
```

### Ví dụ tay tính

5 logs, K=3:

| Log | score | label thật (1=anomaly) |
|-----|-------|------------------------|
| A | 0.95 | 1 |
| B | 0.90 | 0 |
| C | 0.85 | 1 |
| D | 0.20 | 0 |
| E | 0.10 | 0 |

Top 3 score: A, B, C → labels: 1, 0, 1

```
Precision@3 = 2/3 = 66.7%
```

(B trong top 3 nhưng không phải anomaly → false positive)

### Tại sao dùng Precision@K thay vì accuracy?

Log anomaly **rất hiếm** (0.1%-5%). Accuracy misleading:

```
99% log bình thường → model luôn predict "normal" → accuracy 99% nhưng vô dụng
```

Precision@K hỏi: **"Trong những log tôi alert, bao nhiêu % đúng?"**

### Proxy label trong project

HDFS_2k **không có label**. Project dùng:

```
y_proxy = is_rare feature (1 nếu template_count <= 2 trong train)
```

```
Precision@100 (proxy) = 5%
→ Trong top 100 ML score, 5 log có is_rare=1
```

**Hạn chế:** is_rare ≠ anomaly thật. Log is_rare=0 vẫn có thể anomaly. Metric này chỉ để **so sánh tương đối** ML vs baseline khi học tập.

### Metric production nên có

| Metric | Ý nghĩa |
|--------|---------|
| Precision@K | Alert có đáng tin không |
| Recall@K | Bắt được bao nhiêu % anomaly thật |
| F1 | Cân bằng precision/recall |
| Alert volume/day | Ops có chịu nổi không |

Cần label từ incident/ticket hoặc HDFS full labeled dataset.

---

## 4. Model đã train bản chất chứa gì?

File: `models/anomaly_model.joblib`

### Không phải

- Không phải file CSV rules
- Không phải 1 công thức `if template_count <= 2`
- Không phải embedding/neural network weights lớn

### Thực chất: rừng cây quyết định

```
IsolationForest (đã fit)
├── tree 1: các node split (feature_idx, threshold, left, right)
├── tree 2: ...
├── ...
└── tree 100: ...
+ metadata: n_features=63, contamination=0.05, ...
```

Mỗi **node** trong mỗi cây lưu:

| Thành phần | Ví dụ |
|------------|-------|
| `feature` | cột 0 (`template_count`) |
| `threshold` | 150.0 |
| `left`, `right` | con trái/phải |

Khi score 1 log mới:

```
1. Đưa vector X = [254, 0, 64, 1, 0, ...] vào từng cây
2. Mỗi cây: đi xuống theo split cho đến leaf
3. Đếm path length mỗi cây
4. Trung bình → score_samples
5. Normalize → anomaly_score
```

### Kích thước model

100 cây × ~depth 8-12 nodes ≈ vài trăm KB đến vài MB. Nhỏ, deploy dễ.

### Cùng pipeline với feature_builder

Production inference (Part 7):

```
log text
  → parse template (feature_builder từ Part 4)
  → X vector 63 số
  → anomaly_model.joblib (Part 5)
  → score
```

**2 file joblib:**
- `feature_builder.joblib`: log → X
- `anomaly_model.joblib`: X → score

### fit() đã "học" gì?

Không học label. Học **cấu trúc không gian** của data bình thường:

```
"Hầu hết log có template_count 50-300, is_rare=0,
 level_INFO=1, tfidf_block cao..."
```

Điểm nằm ngoài vùng đó → path ngắn → anomaly.

---

## 5. Liên hệ với code project

| Khái niệm | File config/code |
|-----------|------------------|
| n_estimators, contamination | `configs/project.yaml` → `model.*` |
| fit / score | `src/models/train.py` |
| precision@K | `src/models/evaluate.py` |
| pipeline train | `scripts/run_train_part5.py` |
| output | `models/anomaly_model.joblib` |

---

## 6. Câu hỏi thường gặp

**Q: IF có cần scale features không?**  
A: IF split theo ngưỡng, không nhạy như K-Means. Project chưa StandardScaler; có thể thêm ở Part 4 nếu cần.

**Q: Train lại khi có log mới?**  
A: Có. Retrain định kỳ (Part 8) vì pattern "bình thường" drift theo thời gian.

**Q: ML beat baseline v2 chưa?**  
A: Proxy metric: ML ≈ v2 (5% precision@100). Cần label thật để kết luận production.

**Q: Tại sao ML flag nhiều hơn v2 (6.2% vs 1.2%)?**  
A: ML nhìn 63 features; v2 chỉ nhìn template_count + level. ML sensitive hơn, có thể nhiều false positive.

---

## 7. Đọc tiếp

- [LEARNING-part5.md](./LEARNING-part5.md): hướng dẫn chạy Part 5
- [LEARNING-part4.md](./LEARNING-part4.md): X matrix từ đâu ra
- Part 6 (sắp tới): log experiment vào MLflow
