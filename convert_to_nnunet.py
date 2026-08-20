"""Convert the ATLAS/ISLES'26 BIDS tree into an nnU-Net v2 raw dataset.

Keeps the data in native space (no registration). Writes imagesTr/labelsTr, dataset.json,
and a manifest.csv (case_id, site, days_post_stroke, chronicity) used by make_site_splits.py.

Usage:
    python convert_to_nnunet.py /path/to/ATLAS3_Training_Raw /path/to/nnUNet_raw/Dataset001_ISLES2026
"""

import csv
import glob
import json
import os
import shutil
import sys


def find_metadata(anat_dir):
    csvs = glob.glob(os.path.join(anat_dir, "*metadata.csv"))
    if not csvs:
        return {}
    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def convert(atlas_root, out_dir):
    images = os.path.join(out_dir, "imagesTr")
    labels = os.path.join(out_dir, "labelsTr")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)

    t1s = sorted(glob.glob(os.path.join(atlas_root, "*", "sub-*", "ses-*", "anat", "*T1w.nii.gz")))
    manifest = []
    for i, t1 in enumerate(t1s, start=1):
        anat = os.path.dirname(t1)
        masks = glob.glob(os.path.join(anat, "*label-lesion*mask.nii.gz"))
        if not masks:
            continue
        case_id = f"isles_{i:04d}"
        shutil.copy(t1, os.path.join(images, f"{case_id}_0000.nii.gz"))
        shutil.copy(masks[0], os.path.join(labels, f"{case_id}.nii.gz"))
        meta = find_metadata(anat)
        manifest.append({
            "case_id": case_id,
            "site": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(anat)))),
            "days_post_stroke": meta.get("DAYS_POST_STROKE", ""),
            "chronicity": meta.get("CHRONICITY", ""),
        })

    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump({
            "channel_names": {"0": "T1"},
            "labels": {"background": 0, "infarct": 1},
            "numTraining": len(manifest),
            "file_ending": ".nii.gz",
        }, f, indent=2)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "site", "days_post_stroke", "chronicity"])
        w.writeheader()
        w.writerows(manifest)

    print(f"converted {len(manifest)} cases into {out_dir}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
