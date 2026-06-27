"""
Training & Evaluation Script -- HeMIS + SSL Fusion (BraTS 2021)
Original working version that achieved Mean Dice 0.699
"""

import os, json, argparse, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hemis_ssl_model import HeMIS_SSL, HeMIS_SSL_Loss, run_sanity_check
import math
from torch.optim.lr_scheduler import LambdaLR
import multiprocessing
multiprocessing.freeze_support()
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class BraTS2021Dataset(Dataset):
    _MODALITY_ORDER = ['flair', 't1ce', 't1', 't2']

    def __init__(self, data_dir, json_path, val_fold=0, patch_size=128,
                 missing_prob=0.3, augment=True, mode='train',
                 cache_dir=None):
        super().__init__()
        self.data_dir     = data_dir
        self.patch_size   = patch_size
        self.missing_prob = missing_prob if mode == 'train' else 0.0
        self.augment      = augment and (mode == 'train')
        self.mode         = mode
        self.cache_dir    = cache_dir

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"  Cache enabled -> {self.cache_dir}")

        with open(json_path, 'r') as f:
            meta = json.load(f)

        all_entries = meta['training']

        if mode == 'train':
            entries = [e for e in all_entries if e['fold'] != val_fold]
        else:
            entries = [e for e in all_entries if e['fold'] == val_fold]

        self.cases = []

        for entry in entries:
            img_paths = entry['image']
            mod_map = {}

            for path in img_paths:
                for mod in self._MODALITY_ORDER:
                    if path.endswith(f'_{mod}.nii.gz'):
                        mod_map[mod] = os.path.join(data_dir, path)
                        break

            seg_path = os.path.join(data_dir, entry['label'])
            all_paths = list(mod_map.values()) + [seg_path]

            if all(os.path.exists(p) for p in all_paths) and len(mod_map) == 4:
                case_id = os.path.basename(seg_path)
                case_id = case_id.replace('.nii.gz', '')
                case_id = case_id.replace('_seg', '')
                case_id = case_id.replace('_label', '')

                self.cases.append({
                    'id': case_id,
                    'modalities': mod_map,
                    'seg': seg_path
                })

        print(f"  [{mode}] Fold={val_fold} -> {len(self.cases)} cases loaded.")

    def _load_nii(self, path):
        import nibabel as nib
        return nib.load(path).get_fdata(dtype=np.float32)

    def _normalize(self, vol):
        mask = vol > 0
        if mask.sum() > 0:
            p_low  = np.percentile(vol[mask], 0.05)
            p_high = np.percentile(vol[mask], 99.95)
            vol = np.clip(vol, p_low, p_high)
            vol[mask] = (vol[mask] - vol[mask].mean()) / (vol[mask].std() + 1e-8)
        return vol

    def _central_crop(self, vols, seg, size):
        D, H, W = vols[0].shape
        d = max((D - size) // 2, 0)
        h = max((H - size) // 2, 0)
        w = max((W - size) // 2, 0)

        vols = [v[d:d+size, h:h+size, w:w+size] for v in vols]
        seg  = seg[d:d+size, h:h+size, w:w+size]

        return vols, seg

    def _augment(self, vols, seg):
        # Spatial flips
        for axis in range(3):
            if random.random() > 0.5:
                vols = [np.flip(v, axis).copy() for v in vols]
                seg  = np.flip(seg, axis).copy()

        # Intensity augmentation
        for i in range(len(vols)):
            if random.random() > 0.5:
                vols[i] = vols[i] + np.random.uniform(-0.1, 0.1)

            if random.random() > 0.5:
                vols[i] = vols[i] * np.random.uniform(0.9, 1.1)

        return vols, seg

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        filename = self.cases[idx]

        cache_file = None

        if self.cache_dir is not None:
            cache_file = os.path.join(
                self.cache_dir,
                f"{filename['id']}_ps{self.patch_size}.pt"
            )

        # ── Load from cache if available ─────────────────────────
        if cache_file is not None and os.path.exists(cache_file):
            cached = torch.load(
                cache_file,
                map_location='cpu',
                weights_only=False
            )

            mods_tensor = cached['mods']   # [4, D, H, W]
            seg_tensor  = cached['seg']    # [D, H, W]

            mods = [mods_tensor[i].numpy() for i in range(4)]
            seg  = seg_tensor.numpy().astype(np.int64)

        # ── Otherwise load NIfTI, preprocess, and save cache ─────
        else:
            mods = []

            for key in ['flair', 't1ce', 't1', 't2']:
                v = self._normalize(self._load_nii(filename['modalities'][key]))
                mods.append(v)

            seg = self._load_nii(filename['seg']).astype(np.int64)
            seg[seg == 4] = 3

            mods, seg = self._central_crop(mods, seg, self.patch_size)

            if cache_file is not None:
                mods_tensor = torch.from_numpy(
                    np.stack(
                        [v.astype(np.float32) for v in mods],
                        axis=0
                    )
                )

                seg_tensor = torch.from_numpy(seg.astype(np.uint8))

                torch.save(
                    {
                        'mods': mods_tensor,
                        'seg': seg_tensor
                    },
                    cache_file
                )

        # ── Apply augmentation after loading/cache ───────────────
        if self.augment:
            mods, seg = self._augment(mods, seg)

        present = []

        for v in mods:
            if random.random() < self.missing_prob:
                present.append(None)
            else:
                present.append(
                    torch.from_numpy(v.copy()).unsqueeze(0).float()
                )

        if all(m is None for m in present):
            keep = random.randint(0, 3)
            present[keep] = torch.from_numpy(
                mods[keep].copy()
            ).unsqueeze(0).float()

        return present, torch.from_numpy(seg.copy()).long()

# ─────────────────────────────────────────────
# Pretraining
# ─────────────────────────────────────────────
def pretrain(args, model, device):
    print("\n-- SSL Pretraining (Masked Image Modeling) ---------------")
    dataset = BraTS2021Dataset(
        args.data_dir,
        args.json,
        val_fold=args.val_fold,
        patch_size=args.patch_size,
        missing_prob=0.0,
        augment=True,
        mode='train',
        cache_dir=args.cache_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=args.workers > 0
    )

    pretrain_params = list(model.parameters())

    optimizer = optim.AdamW(pretrain_params, lr=args.lr, weight_decay=0.05)

    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=10,
        total_epochs=args.epochs
    )

    start_epoch = getattr(args, "start_epoch", 1)

    # Important: previous best MIM loss was 0.5920
    # This prevents overwriting ssl_best.pth unless the continued run improves it.
    best_loss = float("inf")
    start_epoch = args.start_epoch
    checkpoint = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            print(f"  Optimizer state restored")
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
            print(f"  Scheduler state restored")
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"  Resumed from {args.resume}")
        print(f"  Epoch: {ckpt.get('epoch')}, Best loss: {best_loss:.4f}")
        print(f"  Continuing from epoch {start_epoch} to {args.epochs}")


    else:
        best_loss = float("inf")
        print("  Starting SSL pretraining from scratch.")
        print(f"  Training from epoch {start_epoch} to {args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        good_batches = 0

        for mods, _ in loader:
            mods = [m.to(device, non_blocking=True) if m is not None else None for m in mods]

            optimizer.zero_grad(set_to_none=True)

            # SSL pretraining in FP32, not AMP
            loss = model(mods, ssl_pretrain=True, mask_ratio=args.mask_ratio)

            if not torch.isfinite(loss):
                print("  [BAD LOSS] NaN/Inf detected. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            good_batches += 1

        scheduler.step()

        if good_batches == 0:
            avg_loss = float("inf")
            print(f"  Epoch [{epoch:03d}/{args.epochs}]  MIM Loss: inf  No valid batches")
        else:
            avg_loss = total_loss / good_batches

        if avg_loss < best_loss:
            best_loss = avg_loss

            save_checkpoint({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, "./checkpoints/ssl_best_continued.pth")

            print(f"  Checkpoint saved -> ./checkpoints/ssl_best_continued.pth")
            print(f"  Epoch [{epoch:03d}/{args.epochs}]  MIM Loss: {avg_loss:.4f}  ★ New best continued SSL saved")

        else:
            print(f"  Epoch [{epoch:03d}/{args.epochs}]  MIM Loss: {avg_loss:.4f}  Best: {best_loss:.4f}")

        if epoch % 25 == 0:
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "current_loss": avg_loss,
                    "args": vars(args),
                },
                f"./checkpoints/ssl_ep{epoch:03d}_continued.pth"
            )

            print(f"  Checkpoint saved -> ./checkpoints/ssl_ep{epoch:03d}_continued.pth")


    if os.path.exists("./checkpoints/ssl_best_continued.pth"):
        load_checkpoint(model, "./checkpoints/ssl_best_continued.pth", device)

        save_checkpoint(
            {
                "model_state": model.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            },
            "./checkpoints/ssl_pretrained_continued.pth"
        )

        print("  Continued SSL pretraining done -> ssl_pretrained_continued.pth")
    else:
        print("  Continued SSL pretraining finished, but no new best checkpoint was created.")
        print("  Keep using your previous best SSL checkpoint.")
