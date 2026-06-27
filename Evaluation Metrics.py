# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred_logits, target, num_classes=4, ignore_bg=True):
    pred = torch.argmax(pred_logits, dim=1)
    
    # Per-label scores (NCR, ED, ET)
    scores = []
    for c in range(1, num_classes):
        p = (pred == c).float().reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append((2 * inter / (denom + 1e-5)).item())
    
    # Clinical region scores
    # TC = Tumor Core = NCR(1) + ET(3)
    p_tc = ((pred == 1) | (pred == 3)).float().reshape(-1)
    t_tc = ((target == 1) | (target == 3)).float().reshape(-1)
    inter_tc = (p_tc * t_tc).sum()
    tc = (2 * inter_tc / (p_tc.sum() + t_tc.sum() + 1e-5)).item()
    
    # WT = Whole Tumor = NCR(1) + ED(2) + ET(3)
    p_wt = (pred >= 1).float().reshape(-1)
    t_wt = (target >= 1).float().reshape(-1)
    inter_wt = (p_wt * t_wt).sum()
    wt = (2 * inter_wt / (p_wt.sum() + t_wt.sum() + 1e-5)).item()
    
    # ET = label 3
    et = scores[2]
    
    return scores, et, tc, wt

def collate_fn(batch):
    mod_lists = [item[0] for item in batch]
    segs = torch.stack([item[1] for item in batch])
    n_mod = len(mod_lists[0])
    batched_mods = []
    for m in range(n_mod):
        tensors = [mod_lists[b][m] for b in range(len(batch))]
        if all(t is not None for t in tensors):
            batched_mods.append(torch.stack(tensors))
        elif all(t is None for t in tensors):
            batched_mods.append(None)
        else:
            ref = next(t for t in tensors if t is not None)
            out = [t if t is not None else torch.zeros_like(ref) for t in tensors]
            batched_mods.append(torch.stack(out))
    return batched_mods, segs


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_state  = model.state_dict()
    pretrained   = ckpt['model_state']
    matched  = {k: v for k, v in pretrained.items()
                if k in model_state and v.shape == model_state[k].shape}
    skipped  = [k for k in pretrained.keys()
                if k not in matched]
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"  [INFO] Loaded {len(matched)} layers, skipped {len(skipped)}: {skipped}")
    return ckpt.get('epoch', 0), ckpt.get('best_dice', 0.0)
