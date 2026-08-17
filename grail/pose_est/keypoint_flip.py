"""Left/right label-swap repair for 2D body keypoints.

ViTPose assigns left/right labels independently per frame with no temporal
consistency.  In side views, where one limb occludes the other, the estimator
can swap the two labels mid-sequence and keep them swapped for many frames.

Downstream this is not a cosmetic error.  GENMO fits the 3D body to these
keypoints, so a swap makes the raised leg jump to the other side and back —
which reads as a violent twitch in the reconstruction — and it also mislabels
which limb is which in the exported motion, so the retargeted robot ends up
stepping with the wrong leg.

The repair is a Viterbi pass over a per-frame binary "is this chain swapped"
state.  The signal is strong: real motion moves each joint a few pixels per
frame, while a label swap teleports both joints of a pair across the body.
Limb chains are corrected independently because the estimator flips them
independently (in practice legs flip and arms don't).
"""

import numpy as np

# COCO-17 index pairs, (left, right), grouped into the chains that flip as a unit.
# The hip pair moves very little under a leg swap and so contributes almost
# nothing to the decision, but it is included so the corrected chain stays
# internally consistent with its knee and ankle.
COCO17_CHAINS = {
    "arm": [(5, 6), (7, 8), (9, 10)],
    "leg": [(11, 12), (13, 14), (15, 16)],
}


def _chain_costs(kp, pairs):
    """Per-transition cost of keeping vs. swapping a chain's L/R assignment.

    Each pair contributes the total distance its two joints travelled between
    consecutive frames under the given assignment, weighted by the confidence
    of the joints involved so that occluded or missing detections don't vote.
    """
    xy, conf = kp[..., :2], kp[..., 2]
    prev, cur = slice(0, -1), slice(1, None)

    keep = np.zeros(len(kp) - 1)
    swap = np.zeros(len(kp) - 1)
    for i, j in pairs:
        w = np.minimum(conf[prev, i], conf[cur, i]) * np.minimum(conf[prev, j], conf[cur, j])
        keep += w * (
            np.linalg.norm(xy[cur, i] - xy[prev, i], axis=-1)
            + np.linalg.norm(xy[cur, j] - xy[prev, j], axis=-1)
        )
        swap += w * (
            np.linalg.norm(xy[cur, j] - xy[prev, i], axis=-1)
            + np.linalg.norm(xy[cur, i] - xy[prev, j], axis=-1)
        )
    return keep, swap


def _viterbi_parity(keep, swap, switch_penalty):
    """Cheapest sequence of per-frame parities given keep/swap transition costs.

    State is the parity of frame t relative to the input labels.  Only the
    *relative* parity of consecutive frames selects between `keep` and `swap`,
    so the transition cost depends on the XOR of the two states.
    """
    n = len(keep) + 1
    cost = np.zeros(2)  # cost[s] = best cost of reaching frame t in state s
    back = np.zeros((n, 2), dtype=np.int8)

    for t in range(1, n):
        new = np.empty(2)
        for s in (0, 1):
            # same parity as previous frame => joints kept their labels
            opts = [
                cost[s] + keep[t - 1],
                cost[1 - s] + swap[t - 1] + switch_penalty,
            ]
            k = int(np.argmin(opts))
            new[s] = opts[k]
            back[t, s] = s if k == 0 else 1 - s
        cost = new

    parity = np.zeros(n, dtype=bool)
    s = int(np.argmin(cost))
    for t in range(n - 1, -1, -1):
        parity[t] = bool(s)
        s = int(back[t, s])
    return parity


def fix_lr_swaps(keypoints, chains=None, switch_penalty=30.0):
    """Repair left/right label swaps in a 2D keypoint sequence.

    Args:
        keypoints (np.ndarray | torch.Tensor): (T, 17, 3) COCO-17 keypoints,
            last channel is confidence.
        chains (dict | None): name -> list of (left_idx, right_idx) pairs to
            correct as a unit.  Defaults to :data:`COCO17_CHAINS`.
        switch_penalty (float): confidence-weighted pixel cost charged for
            changing parity between two frames.  Guards against flip-flopping
            on noise; genuine swaps beat it by an order of magnitude.

    Returns:
        (fixed, info): `fixed` is a new array of the same shape and type as the
        input.  `info` maps each chain name to the list of frame indices where
        the corrected parity changes — i.e. where a swap was detected.
    """
    chains = COCO17_CHAINS if chains is None else chains

    is_tensor = hasattr(keypoints, "detach")
    kp = keypoints.detach().cpu().numpy() if is_tensor else np.asarray(keypoints)
    if kp.ndim != 3 or kp.shape[-1] < 3:
        raise ValueError(f"Expected (T, J, 3) keypoints, got {kp.shape}")

    fixed = kp.copy()
    info = {}
    for name, pairs in chains.items():
        if len(kp) < 2:
            info[name] = []
            continue

        keep, swap = _chain_costs(kp, pairs)
        parity = _viterbi_parity(keep, swap, switch_penalty)

        # Viterbi only pins down parity up to a global flip.  Anchor it to the
        # estimator's own majority labelling: a swap that persists for most of
        # the clip is far more likely to be our anchor being wrong than ViTPose
        # being wrong throughout.
        if parity.mean() > 0.5:
            parity = ~parity

        for i, j in pairs:
            fixed[parity, i], fixed[parity, j] = kp[parity, j], kp[parity, i]

        info[name] = (np.flatnonzero(np.diff(parity)) + 1).tolist()

    if is_tensor:
        import torch

        fixed = torch.from_numpy(fixed).to(keypoints.device, keypoints.dtype)
    return fixed, info
