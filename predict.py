"""Ensemble prediction: average three nnU-Net schemes, then post-process.

For every input T1, we run three trainers (default / residual-encoder / Top-K loss),
average their softmax with equal weight, and apply confidence-aware post-processing.
The total number of forward passes matches a single 5-fold model, spread across schemes.

Usage:
    python predict.py --input INPUT_DIR --output OUTPUT_DIR --models NNUNET_RESULTS_DIR
Input images must follow nnU-Net naming: <case>_0000.nii.gz
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from postprocess import confidence_aware_postprocess

# (results subfolder, folds) — 5 forward passes total, spread across schemes for diversity
SCHEMES = [
    ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0, 1)),
    ("nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", (0, 1)),
    ("nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres", (0,)),
]


def _predictor():
    return nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           perform_everything_on_device=True, allow_tqdm=False)


def _foreground_prob(scheme_dir, case, ref_image, ref_shape):
    prob = np.load(scheme_dir / f"{case}.npz")["probabilities"][1].astype(np.float32)
    if prob.shape != ref_shape:  # resample back to the input grid if needed
        seg = sitk.ReadImage(str(scheme_dir / f"{case}.nii.gz"))
        img = sitk.GetImageFromArray(prob)
        img.SetSpacing(seg.GetSpacing())
        img.SetOrigin(seg.GetOrigin())
        img.SetDirection(seg.GetDirection())
        img = sitk.Resample(img, ref_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        prob = sitk.GetArrayFromImage(img).astype(np.float32)
    return prob


def main(input_dir, output_dir, models_root):
    input_dir, output_dir, models_root = Path(input_dir), Path(output_dir), Path(models_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())

    scheme_dirs = []
    for name, folds in SCHEMES:
        predictor = _predictor()
        predictor.initialize_from_trained_model_folder(
            str(models_root / name), use_folds=folds, checkpoint_name="checkpoint_best.pth")
        out = work / name
        predictor.predict_from_files(str(input_dir), str(out), save_probabilities=True, overwrite=True)
        scheme_dirs.append(out)

    for t1 in sorted(input_dir.glob("*_0000.nii.gz")):
        case = t1.name.replace("_0000.nii.gz", "")
        ref = sitk.ReadImage(str(t1))
        ref_shape = sitk.GetArrayFromImage(ref).shape

        ens = np.mean([_foreground_prob(d, case, ref, ref_shape) for d in scheme_dirs], axis=0)
        seg = (ens >= 0.5).astype(np.uint8)
        seg = confidence_aware_postprocess(seg, ens, ref.GetSpacing())

        out_img = sitk.GetImageFromArray(seg)
        out_img.CopyInformation(ref)
        sitk.WriteImage(out_img, str(output_dir / f"{case}.nii.gz"), useCompression=True)
        print(f"{case}: {int(seg.sum())} lesion voxels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", required=True, help="nnUNet_results/Dataset001_ISLES2026 folder")
    a = ap.parse_args()
    main(a.input, a.output, a.models)
