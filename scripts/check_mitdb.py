#!/usr/bin/env python3
"""Check the MIT-BIH download and print AAMI class counts for each split."""

from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import toml

from datasets.mitdb import MITDBClassificationDataset
from utils import dict_to_object


def main():
    config_path = REPO_ROOT / "configs/datasets/mitdb_biomedcoop.toml"
    config = dict_to_object(toml.load(config_path))

    for split in ("train", "val", "test"):
        dataset = MITDBClassificationDataset(config, split)
        counts = Counter(dataset.labels.tolist())
        named = {
            dataset.class_order[index]: counts.get(index, 0)
            for index in range(dataset.n_classes)
        }
        print(
            f"{split:>5}: samples={len(dataset):6d}, "
            f"shape={tuple(dataset.records.shape)}, counts={named}"
        )


if __name__ == "__main__":
    main()
