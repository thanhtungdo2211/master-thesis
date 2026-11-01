"""Loss sanity checks for --loss {sdm,infonce,mpce} (needs torch, no GPU/dataset)."""
import math, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_lora_fedavg.loss import LOSSES, infonce_loss, mpce_loss, sdm_loss

torch.manual_seed(0)
B, D, scale = 16, 32, 50.0
img, txt = torch.randn(B, D), torch.randn(B, D)
uniq = torch.arange(B)                       # every pid distinct -> q = identity
pid = torch.tensor([0, 0, 1, 1] + list(range(2, 14)))  # two positive pairs off-diagonal

# 1. all three run and are finite
for name, fn in LOSSES.items():
    v = fn(img, txt, pid, scale)
    assert torch.isfinite(v), name
print("1. all losses finite: OK")

# 2. with distinct pids, mpce reduces to InfoNCE (up to IRRA's /2 averaging)
a, b = mpce_loss(img, txt, uniq, scale), infonce_loss(img, txt, uniq, scale)
assert torch.allclose(a, 2 * b, atol=1e-4), (a, b)
print("2. mpce == 2 * infonce when every pid is distinct: OK")

# 3. ...whereas SDM does NOT: KL(p||q) with one-hot q = -H(p) + (1 - p_ii) log(1/eps)
i_n, t_n = F.normalize(img, dim=1), F.normalize(txt, dim=1)
want = 0.0
for logits in (scale * i_n @ t_n.t(), scale * t_n @ i_n.t()):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    want += ((p * logp).sum(1) + (1 - p.diagonal()) * math.log(1 / 1e-8)).mean()
got = sdm_loss(img, txt, uniq, scale)
assert torch.allclose(got, want, rtol=1e-3), (got, want)
assert not torch.allclose(got, 2 * infonce_loss(img, txt, uniq, scale), rtol=1e-2)
print(f"3. sdm != infonce for one-hot labels ({got:.3f} vs {2*b:.3f}): OK")

# 4. infonce ignores pid, the identity-aware losses do not
assert torch.allclose(infonce_loss(img, txt, pid, scale), infonce_loss(img, txt, uniq, scale))
for fn in (sdm_loss, mpce_loss):
    assert not torch.allclose(fn(img, txt, pid, scale), fn(img, txt, uniq, scale))
print("4. pid sensitivity (infonce: none, sdm/mpce: yes): OK")

# 5. perfect alignment drives sdm and mpce towards their minimum (0 and H(q))
feat = F.one_hot(uniq, B).float()
assert sdm_loss(feat, feat, uniq, 1000.0) < 1e-3
assert mpce_loss(feat, feat, uniq, 1000.0) < 1e-3
print("5. perfect alignment -> ~0 loss: OK")

print("\nALL LOSS TESTS PASSED")
