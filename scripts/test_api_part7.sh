#!/bin/bash
# Part 7: curl test khi server đang chạy (python scripts/run_api_part7.py)

BASE_URL="http://127.0.0.1:8000"
OUTPUT_FILE="$(dirname "$0")/test_api_part7_output.txt"

exec > >(tee "$OUTPUT_FILE") 2>&1
echo "Generated at: $(date)"
echo ""

echo "=== Case 1: Health check ==="
echo "curl -s $BASE_URL/health"
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== Case 2: Predict log bình thường ==="
echo "curl -s -X POST $BASE_URL/predict -H 'Content-Type: application/json' -d '{...}'"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-06-21T10:00:00Z",
    "level": "INFO",
    "service": "dfs.DataNode",
    "message": "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010"
  }' | python3 -m json.tool

echo ""
echo "=== Case 3: Predict log lạ ==="
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-06-21T10:00:01Z",
    "level": "INFO",
    "service": "dfs.DataNode",
    "message": "Totally unknown event xyz-999-never-seen-before"
  }' | python3 -m json.tool

echo ""
echo "=== Case 4: Invalid payload (missing message) ==="
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-06-21T10:00:00Z", "level": "INFO"}' | python3 -m json.tool

echo ""
echo "Output saved to: $OUTPUT_FILE"
