"""Build a leakage-free 5-fold split: every imaging center is held out in exactly one fold.

Reads the manifest written by convert_to_nnunet.py and writes nnU-Net's splits_final.json
into the preprocessed dataset folder. Random folds mix centers between train and validation
and overestimate performance; grouping by center gives an honest read on unseen sites.

Usage:
    python make_site_splits.py manifest.csv /path/to/nnUNet_preprocessed/Dataset001_.../
"""

import csv
import json
import sys
from collections import defaultdict


def make_splits(manifest_csv, n_folds=5):
    cases_by_site = defaultdict(list)
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            cases_by_site[row["site"]].append(row["case_id"])

    # assign whole sites to folds, always filling the smallest fold first (balances sizes)
    folds = [[] for _ in range(n_folds)]
    for site, cases in sorted(cases_by_site.items(), key=lambda kv: -len(kv[1])):
        smallest = min(range(n_folds), key=lambda k: len(folds[k]))
        folds[smallest].extend(cases)

    splits = []
    for k in range(n_folds):
        val = folds[k]
        train = [c for j in range(n_folds) if j != k for c in folds[j]]
        splits.append({"train": sorted(train), "val": sorted(val)})
    return splits


if __name__ == "__main__":
    manifest_csv, preprocessed_dir = sys.argv[1], sys.argv[2]
    splits = make_splits(manifest_csv)
    out = f"{preprocessed_dir.rstrip('/')}/splits_final.json"
    with open(out, "w") as f:
        json.dump(splits, f, indent=2)
    for k, s in enumerate(splits):
        print(f"fold {k}: {len(s['train'])} train, {len(s['val'])} val")
    print(f"wrote {out}")
