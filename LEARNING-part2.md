# Part 2: Normalize + Template Parsing

## Mục tiêu

Hiểu tại sao cần **chuẩn hóa schema** và **parse template** trước khi train model.

## Kiến thức nền

### Vấn đề message raw

```
Connection refused to postgres:5432
Connection refused to redis:6379
Invalid token for user_id=9912
Invalid token for user_id=8821
```

Mỗi dòng khác nhau → model coi là 4 pattern riêng.

### Sau template parsing

```
Connection refused to <*>
Connection refused to <*>
Invalid token for user_id=<*>
Invalid token for user_id=<*>
```

Chỉ còn 2 pattern → model học tốt hơn.

### Pipeline Part 2

```
raw log (JSONL hoặc CSV)
  ↓ normalize.py      # cùng schema
structured log
  ↓ template.py       # Drain3
log + template + event_id
  ↓ save_jsonl
data/processed/*.jsonl
```

## Cấu trúc Part 2

```
src/
├── ingest/
│   ├── load_logs.py      # + load_hdfs_csv, save_jsonl
│   └── normalize.py      # JSONL + HDFS -> schema thống nhất
└── parse/
    └── template.py       # Drain3 wrapper

scripts/
├── download_hdfs.py      # tải HDFS_2k từ LogHub
├── run_parse_part2.py    # chạy full pipeline Part 2
└── explore_part2.py      # EDA template
```

## Chạy thử

```bash
cd zzz-code/log-anomaly-mlops
source .venv/bin/activate   # hoặc tạo venv mới
pip install -r requirements.txt

python scripts/download_hdfs.py
python scripts/run_parse_part2.py
python scripts/explore_part2.py
```

## Đọc code theo thứ tự

1. `src/ingest/normalize.py` - gom nhiều format về 1 schema
2. `src/parse/template.py` - message -> template
3. `scripts/run_parse_part2.py` - orchestration
4. `data/processed/parsed_sample.jsonl` - output để xem

## Schema sau normalize

```json
{
  "timestamp": "081109 203518",
  "level": "INFO",
  "message": "Receiving block...",
  "service": "dfs.DataNode",
  "trace_id": "123",
  "source": "hdfs",
  "event_template_gt": "Receiving block <*> src: <*> dest: <*>"
}
```

## Schema sau parse (thêm 2 field)

```json
{
  "...": "...",
  "template": "Connection refused to <*>",
  "event_id": "E3"
}
```

## 3 câu tự kiểm tra

1. Tại sao không train trực tiếp trên `message` raw?
2. `normalize.py` giải quyết vấn đề gì?
3. 3 dòng "Connection refused..." sau parse có mấy template?

## Bài tập nhỏ

1. Mở `parsed_sample.jsonl`, tìm 3 dòng Connection refused, xem `template` có giống nhau không
2. Đổi `parser.sim_threshold` trong config (0.4 -> 0.8), chạy lại, đếm unique templates
3. Thêm 1 dòng log mới vào `sample.jsonl`, chạy lại pipeline

## Part 3 (tiếp theo)

Xem [LEARNING-part3.md](./LEARNING-part3.md)
