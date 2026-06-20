# Part 1: Học từ code

## Mục tiêu

Hiểu bài toán log anomaly detection trước khi train model.

## Cấu trúc Part 1

```
log-anomaly-mlops/
├── configs/project.yaml      # "Hợp đồng" project: bài toán, schema, threshold
├── data/raw/sample.jsonl     # 10 dòng log mẫu để học
├── src/
│   ├── config.py             # Load YAML config
│   ├── ingest/load_logs.py   # Đọc + validate JSONL
│   └── baseline/rules.py     # Rule baseline (floor performance)
└── scripts/
    ├── explore_part1.py      # EDA 5 câu hỏi
    └── run_baseline.py       # Chạy baseline, xem miss cases
```

## Chạy thử

```bash
cd zzz-code/log-anomaly-mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/explore_part1.py
python scripts/run_baseline.py
```

## Đọc code theo thứ tự

1. `configs/project.yaml` - định nghĩa bài toán
2. `src/ingest/load_logs.py` - mọi pipeline bắt đầu từ ingest
3. `src/baseline/rules.py` - baseline phải beat được mới deploy ML
4. `scripts/explore_part1.py` - EDA tự động
5. `scripts/run_baseline.py` - xem baseline flag gì

## 3 câu tự kiểm tra

1. Input/output của model là gì?
2. Tại sao anomaly detection phù hợp hơn classification lúc này?
3. Rule baseline miss case nào? (xem output `run_baseline.py`)

## Bài tập nhỏ

1. Thêm 3 dòng log vào `sample.jsonl`, chạy lại 2 script
2. Sửa rule: `WARN` cũng score 0.5, chạy lại baseline
3. Đổi `anomaly_threshold` trong config, quan sát thay đổi

## Part 2 (tiếp theo)

Xem [LEARNING-part2.md](./LEARNING-part2.md)

## Part 3 (tiếp theo)

- EDA sâu trên HDFS
- Baseline v2 dùng template
