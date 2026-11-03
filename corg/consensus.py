"""Hypothesis consensus (Section III-C of the paper)."""
import numpy as np


def iou(a, b):
    """Intersection over union of two boxes (x1, y1, x2, y2)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def consensus(hypotheses, tau=0.5):
    """Consensus box of the valid box hypotheses.

    The medoid is the hypothesis with the highest mean IoU to the others (Eq. 6). The consensus box averages the
    coordinates of the hypotheses whose IoU with the medoid is at least tau (Eq. 7). A single hypothesis is returned as
    it is, and no box is returned when no hypothesis is valid.
    """
    if not hypotheses:
        return None
    if len(hypotheses) == 1:
        return hypotheses[0]
    agreement = [np.mean([iou(b, x) for j, x in enumerate(hypotheses) if j != i]) for i, b in enumerate(hypotheses)]
    medoid = hypotheses[int(np.argmax(agreement))]
    # the medoid always belongs to its own cluster, also when its area is zero
    cluster = [b for b in hypotheses if b is medoid or iou(b, medoid) >= tau]
    return [float(v) for v in np.mean(np.array(cluster), axis=0)]
