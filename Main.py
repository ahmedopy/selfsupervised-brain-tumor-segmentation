
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        choices=['sanity', 'pretrain', 'finetune', 'eval'],
        default='sanity'
    )

    parser.add_argument('--data_dir',       type=str, default='.')
    parser.add_argument('--json',           type=str, default='brats21_folds.json')
    parser.add_argument('--val_fold',       type=int, default=0)
    parser.add_argument('--patch_size',     type=int, default=128)
    parser.add_argument('--batch_size',     type=int, default=1)
    parser.add_argument('--epochs',         type=int, default=100)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--missing_prob',   type=float, default=0.3)
    parser.add_argument('--mask_ratio',     type=float, default=0.75)
    parser.add_argument('--workers',        type=int, default=0)
    parser.add_argument('--cache_dir',      type=str, default=None)
    parser.add_argument('--resume',         type=str, default=None)
    parser.add_argument('--start_epoch',    type=int, default=1)
    parser.add_argument('--ssl_ckpt',       type=str, default=None)
    parser.add_argument('--ckpt',           type=str, default=None)

    parser.add_argument('--base_ch',        type=int, default=32)
    parser.add_argument('--ssl_embed_dim',  type=int, default=384)
    parser.add_argument('--ssl_depth',      type=int, default=4)
    parser.add_argument('--ssl_patch_size', type=int, default=16)

    parser.add_argument('--seed',           type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    if args.mode == 'sanity':
        run_sanity_check()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = HeMIS_SSL(
        num_modalities=4,
        num_classes=4,
        base_ch=args.base_ch,
        ssl_embed_dim=args.ssl_embed_dim,
        ssl_depth=args.ssl_depth,
        ssl_patch_size=args.ssl_patch_size
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total:,}")

    if args.mode == 'pretrain':
        pretrain(args, model, device)

    elif args.mode == 'finetune':
        if args.ssl_ckpt and not args.resume:
            ep, _ = load_checkpoint(model, args.ssl_ckpt, device)
            print(f"  Loaded SSL weights from {args.ssl_ckpt} (epoch {ep})")

        finetune(args, model, device)

    elif args.mode == 'eval':
        if args.ckpt is None:
            raise ValueError("--ckpt required for eval mode")

        ep, best = load_checkpoint(model, args.ckpt, device)
        print(f"  Loaded checkpoint (epoch {ep}, best_dice {best:.4f})")

        evaluate(args, model, device)


if __name__ == "__main__":
    main()
