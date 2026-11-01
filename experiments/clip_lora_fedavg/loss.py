"""SDM loss (Similarity Distribution Matching), following IRRA.

Important for FL: SDM is parameter-free. It only needs the relation
pid_i == pid_j within a batch, NOT a classifier over the full identity set.
That's why it aggregates cleanly under FedAvg, unlike IRRA's ID loss.
"""
import torch
import torch.nn.functional as F


def sdm_loss(image_feat, text_feat, pid, logit_scale, eps=1e-8):
    """
    image_feat, text_feat: [B, D] (not yet normalized)
    pid: [B] long
    logit_scale: scalar = 1 / temperature (0.02 -> 50.0)
    """
    pid = pid.reshape(-1, 1)
    labels = (pid - pid.t() == 0).float()
    # q: ground-truth distribution, split evenly across EVERY positive in the batch
    labels_dist = labels / labels.sum(dim=1, keepdim=True)

    image_norm = image_feat / image_feat.norm(dim=1, keepdim=True)
    text_norm = text_feat / text_feat.norm(dim=1, keepdim=True)

    t2i_cosine = text_norm @ image_norm.t()
    i2t_cosine = t2i_cosine.t()

    i2t_logits = logit_scale * i2t_cosine
    t2i_logits = logit_scale * t2i_cosine

    i2t_pred = F.softmax(i2t_logits, dim=1)
    t2i_pred = F.softmax(t2i_logits, dim=1)

    log_q = torch.log(labels_dist + eps)
    i2t = i2t_pred * (F.log_softmax(i2t_logits, dim=1) - log_q)
    t2i = t2i_pred * (F.log_softmax(t2i_logits, dim=1) - log_q)

    return i2t.sum(dim=1).mean() + t2i.sum(dim=1).mean()


# ---------- ablation losses (--loss), same signature as sdm_loss ----------

def _logits(image_feat, text_feat, logit_scale):
    image_norm = image_feat / image_feat.norm(dim=1, keepdim=True)
    text_norm = text_feat / text_feat.norm(dim=1, keepdim=True)
    t2i_logits = logit_scale * (text_norm @ image_norm.t())
    return t2i_logits.t(), t2i_logits


def infonce_loss(image_feat, text_feat, pid, logit_scale):
    """CLIP's InfoNCE (IRRA's compute_itc): the ONLY positive of row i is column i.

    pid is ignored on purpose -- same-identity pairs off the diagonal are treated as
    negatives, which is exactly the false-negative problem SDM was designed to fix.
    """
    i2t_logits, t2i_logits = _logits(image_feat, text_feat, logit_scale)
    target = torch.arange(i2t_logits.shape[0], device=i2t_logits.device)
    return (F.cross_entropy(i2t_logits, target) + F.cross_entropy(t2i_logits, target)) / 2


def mpce_loss(image_feat, text_feat, pid, logit_scale):
    """Multi-positive cross-entropy: -sum_j q_ij log p_ij = KL(q||p) + H(q).

    Same identity-based q as SDM, but the FORWARD KL. H(q) does not depend on the model,
    so this differs from SDM only in the KL direction -- the ablation that isolates it.
    """
    pid = pid.reshape(-1, 1)
    labels = (pid - pid.t() == 0).float()
    labels_dist = labels / labels.sum(dim=1, keepdim=True)
    i2t_logits, t2i_logits = _logits(image_feat, text_feat, logit_scale)
    i2t = -(labels_dist * F.log_softmax(i2t_logits, dim=1)).sum(dim=1).mean()
    t2i = -(labels_dist * F.log_softmax(t2i_logits, dim=1)).sum(dim=1).mean()
    return i2t + t2i


LOSSES = {"sdm": sdm_loss, "infonce": infonce_loss, "mpce": mpce_loss}
