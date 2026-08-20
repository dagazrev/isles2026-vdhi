"""Confidence-aware post-processing for lesion masks."""

import numpy as np
import SimpleITK as sitk


def confidence_aware_postprocess(seg, prob, spacing_xyz_mm,
                                 min_volume_ml=0.3, min_peak_prob=0.99):
    """Keep a connected component only if it is large OR confidently predicted.

    Removes low-confidence specks (which hurt the lesion-wise metrics) while
    preserving small but confident lesions. Never returns an empty mask when the
    input has foreground.

    seg            : binary array (Z, Y, X)
    prob           : foreground probability, same shape as seg
    spacing_xyz_mm : voxel spacing (x, y, z) in mm
    """
    seg = (seg > 0).astype(np.uint8)
    if seg.sum() == 0:
        return seg

    voxel_ml = float(np.prod(spacing_xyz_mm)) / 1000.0

    cc = sitk.ConnectedComponentImageFilter()
    cc.SetFullyConnected(True)  # 26-connectivity, to match the ISLES'26 evaluator
    labels = sitk.GetArrayFromImage(cc.Execute(sitk.GetImageFromArray(seg)))

    n = int(labels.max())
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        volume_ml = sizes[i] * voxel_ml
        peak = float(prob[labels == i].max())
        keep[i] = volume_ml >= min_volume_ml or peak >= min_peak_prob

    if not keep[1:].any():  # guard: keep the largest component instead of nothing
        keep[int(np.argmax(sizes[1:])) + 1] = True

    return keep[labels].astype(np.uint8)
