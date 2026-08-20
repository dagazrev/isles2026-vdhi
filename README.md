# ISLES'26 — native-space T1w stroke lesion segmentation

A small, reproducible nnU-Net solution: an ensemble of three complementary training
schemes with confidence-aware post-processing. Trained and evaluated in **native space**
(no template registration), with a **leakage-free, site-grouped** cross-validation.

## Method

- **Preprocessing** — nnU-Net defaults: resample to 1×1×1 mm, z-score normalization.
- **Three schemes** (for prediction diversity):
  1. default nnU-Net (Dice + cross-entropy)
  2. residual-encoder nnU-Net
  3. nnU-Net with a Dice + Top-K loss (emphasizes small / hard lesions)
- **Ensemble** — the softmax maps of the three schemes are averaged with equal weight.
- **Post-processing** ([`postprocess.py`](postprocess.py)) — keep a connected component
  only if its volume ≥ 0.3 mL **or** its peak probability ≥ 0.99 (26-connectivity), with a
  guard that never outputs an empty mask.

## Setup

```bash
pip install -r requirements.txt
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...   # nnU-Net folders
```

## Reproduce

```bash
# 1. Convert the BIDS dataset and write a manifest
python convert_to_nnunet.py  /path/to/ATLAS3_Training_Raw  $nnUNet_raw/Dataset001_ISLES2026

# 2. Fingerprint + preprocess
nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_plan_experiment    -d 001 -pl nnUNetPlannerResEncM        # residual-encoder plan

# 3. Site-grouped split (drop-in for nnU-Net)
python make_site_splits.py  $nnUNet_raw/Dataset001_ISLES2026/manifest.csv \
                            $nnUNet_preprocessed/Dataset001_ISLES2026

# 4. Train the three schemes (folds shown are those used at inference)
for f in 0 1; do nnUNetv2_train 001 3d_fullres $f --npz; done                                  # default
for f in 0 1; do nnUNetv2_train 001 3d_fullres $f -p nnUNetResEncUNetMPlans --npz; done         # residual-encoder
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerDiceTopK10Loss --npz                            # Top-K loss

# 5. Predict (ensemble + post-processing)
python predict.py --input INPUT_DIR --output OUTPUT_DIR \
                  --models $nnUNet_results/Dataset001_ISLES2026
```

## Layout

```
convert_to_nnunet.py   BIDS -> nnU-Net raw dataset + manifest.csv
make_site_splits.py    leakage-free, site-grouped 5-fold split
postprocess.py         confidence-aware connected-component filtering
predict.py             three-scheme ensemble inference + post-processing
docker/                Grand Challenge submission container
```

## Notes

- **Why site-grouped folds:** random folds mix imaging centers between train and validation
  and overestimate performance; holding out whole centers estimates generalization honestly.
- **Why the confidence rule:** the challenge scores lesion detection per case, where spurious
  components are costly; keeping only large *or* confident components removes specks while
  preserving small but confident lesions.
- The inference budget is a fixed number of forward passes, so folds are spread across schemes
  rather than using all five folds of each.

## License

MIT
