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
