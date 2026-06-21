#!/usr/bin/env python3
"""
Part 4: Feature engineering pipeline.

Chạy sau Part 2:
    python scripts/run_features_part4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features.build_features import LogFeatureBuilder
from src.features.split import train_test_split_logs
from src.ingest.load_logs import load_jsonl
from src.parse.template import fit_parser_state_from_logs


def main() -> None:
    config = load_config()
    feat_cfg = config["features"]
    parser_cfg = config["parser"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]
    drain3_state_path = ROOT / parser_cfg["state_file"]

    if not hdfs_path.exists():
        print(f"Chưa có {hdfs_path}")
        print("Chạy: python scripts/run_parse_part2.py")
        return

    logs = load_jsonl(hdfs_path)
    if not drain3_state_path.exists():
        print(f"Building Drain3 state: {drain3_state_path}")
        fit_parser_state_from_logs(
            logs,
            drain3_state_path,
            sim_threshold=parser_cfg["sim_threshold"],
            depth=parser_cfg["depth"],
            max_children=parser_cfg["max_children"],
        )

    train_logs, test_logs, train_idx, test_idx = train_test_split_logs(
        logs,
        test_size=feat_cfg["test_size"],
        random_seed=feat_cfg["random_seed"],
    )

    builder = LogFeatureBuilder(
        rare_count_threshold=feat_cfg["rare_count_threshold"],
        tfidf_max_features=feat_cfg["tfidf_max_features"],
    )
    X_train = builder.fit(train_logs).transform(train_logs)
    X_test = builder.transform(test_logs)

    builder_path = ROOT / feat_cfg["builder_file"]
    builder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, builder_path)

    npz_path = ROOT / feat_cfg["features_npz_file"]
    np.savez(
        npz_path,
        X_train=X_train,
        X_test=X_test,
        train_indices=train_idx,
        test_indices=test_idx,
    )

    meta = {
        **builder.describe(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    meta_path = ROOT / feat_cfg["features_meta_file"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("Part 4: Feature Engineering")
    print("=" * 60)
    print(f"Train logs: {len(train_logs)} | Test logs: {len(test_logs)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Features: {len(builder.feature_names_)}")
    print(f"\nSaved builder: {builder_path}")
    print(f"Saved arrays:  {npz_path}")
    print(f"Saved meta:    {meta_path}")
    print("\nFeature groups:")
    print(f"  numeric:     template_count, is_rare, message_len")
    print(f"  level:       4 one-hot")
    print(f"  service:     {len(builder.service_vocab_) + 1} one-hot")
    print(f"  tfidf:       up to {feat_cfg['tfidf_max_features']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
