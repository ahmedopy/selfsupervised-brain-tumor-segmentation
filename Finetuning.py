
# ─────────────────────────────────────────────
# Finetuning
# ─────────────────────────────────────────────
def finetune(args, model, device):
    print("\n-- Segmentation Finetuning ------------------------------")

    train_ds = BraTS2021Dataset(
        args.data_dir,
        args.json,
        val_fold=args.val_fold,
        patch_size=args.patch_size,
        missing_prob=args.missing_prob,
        mode='train',
        cache_dir=args.cache_dir
    )
    val_ds = BraTS2021Dataset(
        args.data_dir,
        args.json,
        val_fold=args.val_fold,
        patch_size=args.patch_size,
        missing_prob=0.0,
        augment=False,
        mode='val',
        cache_dir=args.cache_dir
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_fn,
                            pin_memory=True,
                            prefetch_factor=4,
                            persistent_workers=args.workers > 0)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn,
                            pin_memory=True,
                            prefetch_factor=4,
                            persistent_workers=args.workers > 0)

    criterion = HeMIS_SSL_Loss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=7, total_epochs=args.epochs)
    scaler    = GradScaler('cuda')
    best_dice = 0.0
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice   = ckpt.get('best_dice', 0.0)
        print(f"  Resumed from {args.resume} "
              f"(epoch {start_epoch-1}, best_dice {best_dice:.4f})")

    print(f"  Training epochs {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for mods, seg in train_loader:
            mods = [m.to(device) if m is not None else None for m in mods]
            seg  = seg.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(mods)
                loss = criterion(logits, seg)
            # OUTSIDE autocast:
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        avg_loss = train_loss / max(len(train_loader), 1)

        if epoch % 1 == 0 or epoch == args.epochs:
            model.eval()
            val_dices = []
            val_et, val_tc, val_wt = [], [], []

            with torch.no_grad():
                for mods, seg in val_loader:
                    mods = [m.to(device) if m is not None else None for m in mods]
                    seg  = seg.to(device)
                    with autocast('cuda'):
                        logits = model(mods)
                    scores, et, tc, wt = dice_score(logits, seg)
                    val_dices.append(scores)
                    val_et.append(et)
                    val_tc.append(tc)
                    val_wt.append(wt)

            mean_dices   = np.mean(val_dices, axis=0)
            mean_overall = mean_dices.mean()
            print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
                f"TrainLoss: {avg_loss:.4f}  "
                f"Val Dice -> NCR: {mean_dices[0]:.3f}  "
                f"ED: {mean_dices[1]:.3f}  ET(label): {mean_dices[2]:.3f}  "
                f"Mean: {mean_overall:.3f}\n"
                f"  Clinical -> ET: {np.mean(val_et):.3f}  "
                f"TC: {np.mean(val_tc):.3f}  WT: {np.mean(val_wt):.3f}")

            if mean_overall > best_dice:
                best_dice = mean_overall
                save_checkpoint(
                    {'model_state': model.state_dict(),
                     'epoch': epoch, 'best_dice': best_dice},
                    "./checkpoints/best_model.pth")
        else:
            print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
                  f"TrainLoss: {avg_loss:.4f}")

    print(f"\n  Best Val Mean Dice: {best_dice:.4f}")
