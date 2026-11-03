"""Consensus-guided refinement with the geometric guard (Section III-D of the paper)."""
import numpy as np

from .consensus import iou


def mask_box(mask):
    """Tight bounding box of a binary mask, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def guard(consensus_box, point, refined_box):
    """Accept the refined box only if it exists, the target point lies inside the consensus box, and the two boxes
    overlap (Eq. 11). Otherwise the consensus box is kept."""
    b, p = consensus_box, point
    if b is None or p is None or refined_box is None:
        return b
    inside = b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
    return refined_box if inside and iou(refined_box, b) > 0 else b


class Refiner:
    """Frozen SAM 2.1 prompted with the consensus box and the target point (multimask output, top-scoring mask)."""

    def __init__(self, checkpoint, config="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
        self._image = None

    def set_image(self, image, key=None):
        """image: PIL RGB image. key: identifier that avoids re-encoding the same image for several queries."""
        if key is None or key != self._image:
            self.predictor.set_image(np.array(image))
            self._image = key

    def refined_box(self, consensus_box, point):
        """Box of the top-scoring SAM mask for the box and point prompts, or None."""
        import torch
        if consensus_box is None or point is None:
            return None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self.predictor.predict(point_coords=np.array([point]), point_labels=np.array([1]),
                                                      box=np.array(consensus_box)[None], multimask_output=True)
        return mask_box(masks[int(np.argmax(scores))] > 0)

    def __call__(self, consensus_box, point):
        """Final box: the refined box if the geometric guard accepts it, otherwise the consensus box."""
        return guard(consensus_box, point, self.refined_box(consensus_box, point))
