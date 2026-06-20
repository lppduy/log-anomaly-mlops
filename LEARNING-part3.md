# Part 3: EDA sâu + Baseline v2

## Mục tiêu

1. Hiểu **phân phối template** (frequency, rare, Pareto)
2. So sánh **baseline v1 vs v2** để thấy giá trị của template feature

## Kiến thức nền

### Template frequency

```
Template A: 310 lần  (phổ biến, bình thường)
Template B: 280 lần
Template C: 1 lần    (hiếm, đáng nghi)
```

Trong anomaly detection unsupervised, **template hiếm** là signal mạnh.

### Pareto effect

Top 5 templates có thể chiếm 70-80% log. Phần còn lại là long tail (nhiều template ít gặp).

### Baseline v1 vs v2

| | v1 (Part 1) | v2 (Part 3) |
|---|-------------|-------------|
| Signal | `level == ERROR` | template count + level |
| Bắt WARN hiếm | Không | Có |
| False positive ERROR | Cao | Giảm (template phổ biến) |

## Cấu trúc Part 3

```
src/
├── eda/
│   └── template_stats.py     # thống kê template
└── baseline/
    └── template_rules.py     # baseline v2 + compare

scripts/
├── explore_part3.py          # EDA frequency
└── run_baseline_v2.py        # v1 vs v2
```

## Chạy thử

```bash
python scripts/run_parse_part2.py   # nếu chưa có processed
python scripts/explore_part3.py
python scripts/run_baseline_v2.py
```

## Đọc code theo thứ tự

1. `src/eda/template_stats.py` - compute + format stats
2. `src/baseline/template_rules.py` - rule v2 + compare
3. `scripts/explore_part3.py` - in report
4. `scripts/run_baseline_v2.py` - xem disagree cases

## Baseline v2 logic

```python
if template_count <= 2:   score = 0.85  # rare
elif level == ERROR:      score = 0.65  # common + error
else:                     score = 0.10  # normal
```

## 3 câu tự kiểm tra

1. Tại sao template hiếm có thể là anomaly?
2. v2 bắt được case nào mà v1 miss? (gợi ý: Slow query WARN)
3. `template_count` tính từ đâu? (cùng dataset, Part 5 sẽ tách train/test)

## Bài tập nhỏ

1. Đổi `baseline_v2.rare_count_threshold` từ 2 -> 5, chạy lại `run_baseline_v2.py`
2. Trong `explore_part3.py` output, đếm bao nhiêu rare templates trên HDFS
3. Thêm 1 dòng INFO với message hoàn toàn mới vào sample, parse lại, xem v2 flag không

## Lưu ý học tập

`template_count` hiện tính trên **cùng file** (có data leakage). Part 4-5 sẽ tách:
- Train set -> học template frequency
- Test set -> score anomaly

## Part 4 (tiếp theo)

- Feature engineering: TF-IDF template, level encoding, service counts
- Chuẩn bị matrix X cho sklearn
