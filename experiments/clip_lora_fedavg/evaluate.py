"""Text-to-image retrieval evaluation: R@1/5/10, mAP, mINP.

Evaluates the GLOBAL model on the centralized test set (200 identities),
never per-client.
"""
import torch
from torch.utils.data import DataLoader

from .data import EvalImageDataset, EvalTextDataset


@torch.no_grad()
def extract_features(model, test_items, root, transform, tokenizer, text_len,
                     device, batch_size=128, num_workers=2):
    model.eval()

    img_ds = EvalImageDataset(test_items, root, transform)
    txt_ds = EvalTextDataset(test_items, tokenizer, text_len)
    img_dl = DataLoader(img_ds, batch_size=batch_size, num_workers=num_workers)
    txt_dl = DataLoader(txt_ds, batch_size=batch_size, num_workers=num_workers)

    ifeats, ipids = [], []
    for b in img_dl:
        f = model.get_image_features(pixel_values=b["pixel_values"].to(device))
        ifeats.append(F_normalize(f).cpu())
        ipids.append(b["pid"])

    tfeats, tpids = [], []
    for b in txt_dl:
        f = model.get_text_features(
            input_ids=b["input_ids"].to(device),
            attention_mask=b["attention_mask"].to(device),
        )
        tfeats.append(F_normalize(f).cpu())
        tpids.append(b["pid"])

    return (torch.cat(ifeats), torch.cat(ipids),
            torch.cat(tfeats), torch.cat(tpids))


def F_normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def compute_metrics(text_feat, text_pid, img_feat, img_pid):
    """Query = text, gallery = image."""
    sim = text_feat.float() @ img_feat.float().t()          # [Nt, Ni]
    idx = sim.argsort(dim=1, descending=True)
    matches = (img_pid[idx] == text_pid.view(-1, 1)).float()  # [Nt, Ni]

    n_query, n_gallery = matches.shape

    # CMC
    cmc = matches.cumsum(dim=1)
    cmc[cmc > 1] = 1
    recalls = {f"R@{k}": (cmc[:, k - 1].mean() * 100).item() for k in (1, 5, 10)}

    # mAP
    n_rel = matches.sum(dim=1)
    tmp = matches.cumsum(dim=1) / torch.arange(1, n_gallery + 1).float().view(1, -1)
    ap = (tmp * matches).sum(dim=1) / n_rel.clamp(min=1)
    recalls["mAP"] = (ap.mean() * 100).item()

    # mINP: rank of the hardest positive
    inp = []
    for i in range(n_query):
        pos = torch.nonzero(matches[i], as_tuple=False)
        if pos.numel() == 0:
            continue
        hardest = pos[-1].item()
        inp.append(n_rel[i].item() / (hardest + 1))
    recalls["mINP"] = (sum(inp) / len(inp) * 100) if inp else 0.0

    return recalls


@torch.no_grad()
def evaluate(model, test_items, root, transform, tokenizer, text_len, device,
             batch_size=128, num_workers=2):
    i_f, i_p, t_f, t_p = extract_features(
        model, test_items, root, transform, tokenizer, text_len,
        device, batch_size, num_workers
    )
    return compute_metrics(t_f, t_p, i_f, i_p)
