"""ISLES 2026: diverse 3-scheme softmax ensemble on native T1w, budget-matched.

Grand Challenge interface (interf0):
  Inputs:  /input/images/t1-brain-mri  +  /input/stroke-metadata.json
  Outputs: /output/images/stroke-lesion-segmentation
           /output/images/lesion-probability-map

MAPPING-style fusion (ATLAS-R2 winner recipe): average the softmax of three
different nnU-Net training schemes, then confidence-aware post-processing.
To respect the T4 ~7 min/case budget we spend the SAME number of forward passes as
the plain 5-fold baseline (5), but spread across schemes for diversity:
    plain  nnUNetPlans           folds 0,1
    ResEnc nnUNetResEncUNetMPlans folds 0,1
    TopK   DiceTopK10Loss         fold  0
Each scheme's folds are averaged, then the three schemes are averaged with EQUAL
weight (matches how the ensemble was validated on the honest OOF, where it beat plain
on the per-case rank metric: mean-rank 1.43 vs 1.58).

Model tarball at /opt/ml/model. Layout:
  /opt/ml/model/nnUNet_results/Dataset001_ISLES2026RawT1/<trainer__plans__3d_fullres>/
      {plans,dataset,dataset_fingerprint}.json + fold_*/checkpoint_best.pth
"""

from __future__ import annotations

import glob
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_ROOT = Path("/opt/ml/model") / "nnUNet_results" / "Dataset001_ISLES2026RawT1"

# (trainer__plans__3d_fullres dirname, folds to use, equal ensemble weight) — 5 passes total.
ENSEMBLE = [
    ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0, 1)),
    ("nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", (0, 1)),
    ("nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres", (0,)),
]

# Confidence-aware post-processing (26-connectivity, matches the panoptica evaluator).
# Keep a predicted component iff volume_ml >= MIN_LESION_ML OR peak_prob >= MIN_PEAK_PROB;
# empty-mask guard keeps the largest component. Prob map left continuous (PR-AUC safe).
MIN_LESION_ML = 0.3
MIN_PEAK_PROB = 0.99


def _show_torch_cuda_info() -> None:
    print("=+=" * 10)
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device_count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}")
    print("=+=" * 10)


def _pick_device() -> torch.device:
    """CUDA when kernels actually run; CPU fallback for local Blackwell smoke only."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.zeros(1, device="cuda"); y = x + 1; torch.cuda.synchronize(); del x, y
        return torch.device("cuda")
    except Exception as exc:  # noqa: BLE001
        print(f"[isles-ensemble] CUDA probe failed ({exc}); using CPU for this host")
        return torch.device("cpu")


def _make_predictor(device):
    return nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"), device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )


def init_model():
    _show_torch_cuda_info()
    device = _pick_device()
    print(f"[isles-ensemble] using device={device}")
    predictors = []
    for name, folds in ENSEMBLE:
        mdir = MODEL_ROOT / name
        if not mdir.is_dir():
            raise FileNotFoundError(f"Missing model folder: {mdir}")
        p = _make_predictor(device)
        p.initialize_from_trained_model_folder(str(mdir), use_folds=folds, checkpoint_name="checkpoint_best.pth")
        predictors.append((name, folds, p))
        print(f"[isles-ensemble] loaded {name} folds {folds}")
    return predictors


def run(model):
    handler = {("stroke-metadata", "t1-brain-mri"): interf0_handler}[get_interface_key()]
    return handler(model)


def _read_fg_prob(out_dir: Path, t1_image, t1_data):
    """Foreground probability from a predictor's saved softmax, resampled to input geometry."""
    npzs = sorted(out_dir.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No probabilities npz under {out_dir}: {list(out_dir.iterdir())}")
    data = np.load(str(npzs[0]))
    key = "probabilities" if "probabilities" in data else ("softmax" if "softmax" in data else list(data.keys())[0])
    probs = data[key]
    prob = (probs[1] if probs.shape[0] > 1 else probs[0]).astype(np.float32) if probs.ndim == 4 else probs.astype(np.float32)
    if prob.shape != t1_data.shape:
        segs = sorted(out_dir.glob("*.nii.gz"))
        ref = sitk.ReadImage(str(segs[0]))
        prob_img = sitk.GetImageFromArray(prob)
        prob_img.SetSpacing(ref.GetSpacing()); prob_img.SetOrigin(ref.GetOrigin()); prob_img.SetDirection(ref.GetDirection())
        prob_img = sitk.Resample(prob_img, t1_image, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        prob = sitk.GetArrayFromImage(prob_img).astype(np.float32)
    return prob


def confidence_aware_postproc(seg: np.ndarray, prob: np.ndarray, spacing_xyz,
                              min_volume_ml: float, min_peak_prob: float) -> np.ndarray:
    """Keep a component iff physically large OR confidently predicted (26-conn). Empty-guard."""
    seg = (seg > 0).astype(np.uint8)
    if seg.sum() == 0:
        return seg
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    ccf = sitk.ConnectedComponentImageFilter(); ccf.SetFullyConnected(True)     # 26-conn
    lab = sitk.GetArrayFromImage(ccf.Execute(sitk.GetImageFromArray(seg)))
    n = int(lab.max())
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        vol_ml = float(sizes[i]) * vox_ml
        peak = float(prob[lab == i].max()) if prob is not None else 1.0
        keep[i] = (vol_ml >= min_volume_ml) or (peak >= min_peak_prob)
    if not keep[1:].any():
        keep[int(np.argmax(sizes[1:])) + 1] = True
    cleaned = keep[lab].astype(np.uint8)
    print(f"[isles-ensemble] conf-aware PP: {n} comps -> {int(keep[1:].sum())} kept, "
          f"fg {int(seg.sum())}->{int(cleaned.sum())}")
    return cleaned


def interf0_handler(predictors):
    t1_image, t1_data = load_image_file_as_array_and_image(location=INPUT_PATH / "images/t1-brain-mri")
    meta = load_json_file(location=INPUT_PATH / "stroke-metadata.json")
    print("Loaded Stroke Metadata:", json.dumps(meta))

    work = Path(tempfile.mkdtemp(prefix="isles_ens_", dir="/tmp"))
    in_dir = work / "input"; in_dir.mkdir()
    case_img = in_dir / "case_0000.nii.gz"
    sitk.WriteImage(t1_image, str(case_img))
    try:
        fg_probs = []
        for k, (name, folds, predictor) in enumerate(predictors):
            out_dir = work / f"out_{k}"; out_dir.mkdir()
            predictor.predict_from_files(
                [[str(case_img)]], str(out_dir), save_probabilities=True, overwrite=True,
                num_processes_preprocessing=1, num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0,
            )
            prob = _read_fg_prob(out_dir, t1_image, t1_data)
            fg_probs.append(prob)
            print(f"[isles-ensemble] {name} folds {folds}: prob_max={float(prob.max()):.4f}")

        ens = np.mean(np.stack(fg_probs, axis=0), axis=0).astype(np.float32)   # equal weight per scheme
        seg = (ens >= 0.5).astype(np.uint8)
        seg = confidence_aware_postproc(seg, ens, t1_image.GetSpacing(), MIN_LESION_ML, MIN_PEAK_PROB)

        write_array_as_image_file(location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
                                  array=seg, reference_image=t1_image)
        write_array_as_image_file(location=OUTPUT_PATH / "images/lesion-probability-map",
                                  array=ens.astype(np.float32), reference_image=t1_image)
        print(f"[isles-ensemble] wrote seg fg={int(seg.sum())} prob_max={float(ens.max()):.4f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["socket"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array_and_image(*, location):
    files = glob.glob(str(location / "*.mha")) + glob.glob(str(location / "*.nii.gz")) + glob.glob(str(location / "*.nii"))
    if not files:
        raise FileNotFoundError(f"No valid image file found in {location}")
    image = sitk.ReadImage(files[0])
    return image, sitk.GetArrayFromImage(image)


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(image, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
