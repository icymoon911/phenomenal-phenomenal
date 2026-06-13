import numpy

from openalea.phenomenal.multi_view_reconstruction import project_voxel_centers_on_image


def segmentation_metrics(img_ref, img_src):
    """
    Compute TP, FP, FN, TN + IoU + Dice between two binary images.

    Parameters
    ----------
    img_ref : array-like
        Ground truth (reference)
    img_src : array-like
        Prediction

    Returns
    -------
    dict with:
        TP, FP, FN, TN (counts)
        IoU
        Dice
    """

    # Convert to boolean (robust to 0/1, 0/255, etc.)
    ref = numpy.asarray(img_ref).astype(bool)
    src = numpy.asarray(img_src).astype(bool)

    if ref.shape != src.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {src.shape}")

    # Confusion matrix components
    TP = numpy.logical_and(ref, src).sum()
    FP = numpy.logical_and(~ref, src).sum()
    FN = numpy.logical_and(ref, ~src).sum()
    TN = numpy.logical_and(~ref, ~src).sum()

    # Metrics (safe divisions)
    union = TP + FP + FN
    iou = TP / union if union != 0 else 1.0

    denom_dice = 2 * TP + FP + FN
    dice = (2 * TP) / denom_dice if denom_dice != 0 else 1.0

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "IoU": iou,
        "Dice": dice,
    }


def reconstruction_metrics(voxels_grid, image_views):
    """
    Compute mean metrics over all views.
    """

    all_metrics = []

    for image_view in image_views.values():
        img_src = project_voxel_centers_on_image(
            voxels_grid.voxels_position,
            voxels_grid.voxels_size,
            image_view.image.shape,
            image_view.projection,
        )

        metrics = segmentation_metrics(image_view.image, img_src)
        all_metrics.append(metrics)

    # Aggregate (mean over views)
    keys = all_metrics[0].keys()
    mean_metrics = {
        k: numpy.mean([m[k] for m in all_metrics]) for k in keys
    }

    return mean_metrics
