# ─────────────────────────────────────────────
# Evaluation - all 15 modality combinations
# + Segmentation visualization
# + All-combination visualization
# + Uncertainty calibration plot
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Evaluation Visualization Helpers
# ─────────────────────────────────────────────

MODALITY_COMBOS = [
    ([0],       "F"),
    ([1],       "T1c"),
    ([2],       "T1"),
    ([3],       "T2"),
    ([0, 1],     "F+T1c"),
    ([0, 2],     "F+T1"),
    ([0, 3],     "F+T2"),
    ([1, 2],     "T1c+T1"),
    ([1, 3],     "T1c+T2"),
    ([2, 3],     "T1+T2"),
    ([0, 1, 2],   "F+T1c+T1"),
    ([0, 1, 3],   "F+T1c+T2"),
    ([0, 2, 3],   "F+T1+T2"),
    ([1, 2, 3],   "T1c+T1+T2"),
    ([0, 1, 2, 3], "All 4"),
]


SEG_COLORS = np.array([
    [0,   0,   0  ],   # BG  — black
    [255, 0,   0  ],   # NCR — red
    [0,   255, 0  ],   # ED  — green
    [0,   0,   255],   # ET  — blue
], dtype=np.uint8)


def seg_to_rgb(seg_map):
    """
    Convert [H, W] integer segmentation map to RGB image.
    0 = background, 1 = NCR, 2 = ED, 3 = ET
    """
    return SEG_COLORS[seg_map]


def get_middle_slice(volume_3d):
    """
    Return middle axial slice from a 3D volume.
    """
    mid = volume_3d.shape[0] // 2
    return volume_3d[mid]


def save_segmentation_figure(
    case_idx,
    flair_slice,
    gt_slice,
    pred_all4,
    pred_flair_only,
    var_all4,
    var_flair,
    dice_all4,
    dice_flair,
    save_dir
):
    """
    Save enhanced 4-panel figure with semi-transparent overlays on anatomical scans.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor='black')

    # fig.suptitle(
    #     f"Case {case_idx + 1:03d} — Qualitative Segmentation Comparison",
    #     fontsize=16,
    #     fontweight='bold',
    #     color='white',
    #     y=0.96
    # )

    # Custom translucent colormap to layer labels on top of grayscale structural context
    from matplotlib.colors import ListedColormap
    overlay_cmap = ListedColormap([
        [0.0, 0.0, 0.0, 0.0],  # Background (transparent)
        [1.0, 0.1, 0.1, 0.6],  # NCR: Rich Red with 0.6 opacity
        [0.1, 0.8, 0.1, 0.45], # ED: Vivid Green with 0.45 opacity
        [0.0, 0.4, 1.0, 0.65]  # ET: Vibrant Blue with 0.65 opacity
    ])

    # Panel 1: Original FLAIR structural layout
    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title("FLAIR Input Volume", fontsize=12, color='white', pad=10)
    axes[0].axis('off')

    # Panel 2: Ground Truth overlay mapping
    axes[1].imshow(flair_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap=overlay_cmap, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth Labels", fontsize=12, color='white', pad=10)
    axes[1].axis('off')

    # Panel 3: Multimodal Prediction (All 4)
    axes[2].imshow(flair_slice, cmap='gray')
    axes[2].imshow(pred_all4, cmap=overlay_cmap, vmin=0, vmax=3)
    axes[2].set_title(
        f"Prediction: All 4 Modalities\n"
        f"Dice: {dice_all4:.3f} | Var: {var_all4:.4f}",
        fontsize=11,
        color='#4AD66D',
        pad=10
    )
    axes[2].axis('off')

    # Panel 4: Single Modality Prediction (FLAIR Only)
    axes[3].imshow(flair_slice, cmap='gray')
    axes[3].imshow(pred_flair_only, cmap=overlay_cmap, vmin=0, vmax=3)
    axes[3].set_title(
        f"Prediction: FLAIR Only\n"
        f"Dice: {dice_flair:.3f} | Var: {var_flair:.4f}",
        fontsize=11,
        color='#E63946',
        pad=10
    )
    axes[3].axis('off')

    legend_patches = [
        mpatches.Patch(color='#1A1A1A', label='Background'),
        mpatches.Patch(color=[1.0, 0.1, 0.1],   label='NCR (Necrotic Core)'),
        mpatches.Patch(color=[0.1, 0.8, 0.1],   label='ED (Edema)'),
        mpatches.Patch(color=[0.0, 0.4, 1.0],   label='ET (Enhancing Tumor)'),
    ]

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=4,
        fontsize=11,
        labelcolor='white',
        facecolor='#111111',
        edgecolor='none',
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"case_{case_idx + 1:03d}_segmentation.png"
    )

    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    return save_path


def save_all_combos_figure(
    case_idx,
    flair_slice,
    gt_slice,
    combo_pred_dict,
    save_dir
):
    """
    Save enhanced 15-modality structural compilation matrix matching professional dark-themes.
    """
    fig, axes = plt.subplots(3, 6, figsize=(24, 13), facecolor='black')
    axes = axes.flatten()

    # fig.suptitle(
    #     # f"Case {case_idx + 1:03d} — Segmentation Across All 15 Modality Combinations",
    #     fontsize=18,
    #     fontweight='bold',
    #     color='white',
    #     y=0.96
    # )

    from matplotlib.colors import ListedColormap
    overlay_cmap = ListedColormap([
        [0.0, 0.0, 0.0, 0.0],  # BG
        [1.0, 0.1, 0.1, 0.55], # NCR
        [0.1, 0.8, 0.1, 0.4],  # ED
        [0.0, 0.4, 1.0, 0.6]   # ET
    ])

    # Grounding columns
    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title("FLAIR Input Volume", fontsize=11, color='white', pad=8)
    axes[0].axis('off')

    axes[1].imshow(flair_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap=overlay_cmap, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth Labels", fontsize=11, color='white', pad=8)
    axes[1].axis('off')

    panel_idx = 2

    for present_ids, combo_name in MODALITY_COMBOS:
        if combo_name not in combo_pred_dict:
            continue

        pred_slice, dice_val, var_val = combo_pred_dict[combo_name]

        axes[panel_idx].imshow(flair_slice, cmap='gray')
        axes[panel_idx].imshow(pred_slice, cmap=overlay_cmap, vmin=0, vmax=3)
        axes[panel_idx].set_title(
            f"{combo_name}\nDice: {dice_val:.3f} | Var: {var_val:.4f}",
            fontsize=10,
            color='#4AD66D' if combo_name == "All 4" else 'white',
            pad=6
        )
        axes[panel_idx].axis('off')

        panel_idx += 1

    for j in range(panel_idx, len(axes)):
        axes[j].axis('off')

    legend_patches = [
        mpatches.Patch(color='#1A1A1A', label='Background'),
        mpatches.Patch(color=[1.0, 0.1, 0.1],   label='NCR (Necrotic Core)'),
        mpatches.Patch(color=[0.1, 0.8, 0.1],   label='ED (Edema)'),
        mpatches.Patch(color=[0.0, 0.4, 1.0],   label='ET (Enhancing Tumor)'),
    ]

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=4,
        fontsize=12,
        labelcolor='white',
        facecolor='#111111',
        edgecolor='none',
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"case_{case_idx + 1:03d}_all_combinations.png"
    )

    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    return save_path


# ─────────────────────────────────────────────
# Evaluation - visualization only
# ─────────────────────────────────────────────

def evaluate(args, model, device):
    print("\n-- Evaluation Visualization Only -------------------------")
    print("  This eval saves only segmentation visualization figures.")
    print("  Use uncertainty_eval.py for CSV tables, scatter plots, bar charts, and heatmaps.")

    viz_dir = './evaluation_viz'
    os.makedirs(viz_dir, exist_ok=True)

    test_ds = BraTS2021Dataset(
        args.data_dir,
        args.json,
        val_fold=args.val_fold,
        patch_size=args.patch_size,
        missing_prob=0.0,
        augment=False,
        mode='val',
        cache_dir=args.cache_dir
    )

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "collate_fn": collate_fn,
        "pin_memory": (device.type == "cuda"),
    }

    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(test_ds, **loader_kwargs)

    VIZ_CASES_LIMIT = 5

    print(f"  Loading first {VIZ_CASES_LIMIT} validation cases for visualization...")

    all_cases = []
    model.eval()

    with torch.no_grad():
        for case_idx, (mods, seg) in enumerate(loader):
            if case_idx >= VIZ_CASES_LIMIT:
                break

            all_cases.append((
                [m.to(device, non_blocking=True) for m in mods],
                seg.to(device, non_blocking=True)
            ))

    VIZ_CASES = len(all_cases)

    print(f"  Loaded {VIZ_CASES} visualization cases.\n")

    if VIZ_CASES == 0:
        print("  No validation cases found. Nothing to visualize.")
        return

    viz_preds_all4 = {}
    viz_preds_flair_only = {}
    viz_preds_by_combo = {i: {} for i in range(VIZ_CASES)}

    print(f"  Evaluating all 15 modality combinations for {VIZ_CASES} visualization cases...")

    with torch.no_grad():
        for combo_idx, (present_ids, combo_name) in enumerate(MODALITY_COMBOS, start=1):
            print(f"    [{combo_idx:02d}/{len(MODALITY_COMBOS)}] Processing {combo_name}")

            for case_idx, (full_mods, seg) in enumerate(all_cases):
                mods = [
                    full_mods[i] if i in present_ids else None
                    for i in range(4)
                ]

                with autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(mods)

                scores, et, tc, wt = dice_score(logits, seg)
                var = model.get_hemis_variance(mods)

                pred_map = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_slice = get_middle_slice(pred_map)
                mean_dice = float(np.mean(scores))

                viz_preds_by_combo[case_idx][combo_name] = (
                    pred_slice,
                    mean_dice,
                    var
                )

                if present_ids == [0, 1, 2, 3]:
                    viz_preds_all4[case_idx] = (
                        pred_slice,
                        mean_dice,
                        var
                    )

                elif present_ids == [0]:
                    viz_preds_flair_only[case_idx] = (
                        pred_slice,
                        mean_dice,
                        var
                    )

    print(f"\n  -- Saving original 4-panel segmentation visualizations ({VIZ_CASES} cases) --")

    for case_idx in range(VIZ_CASES):
        if case_idx not in viz_preds_all4 or case_idx not in viz_preds_flair_only:
            print(f"     Skipped case {case_idx + 1:03d}: missing All-4 or FLAIR-only prediction")
            continue

        flair_vol = all_cases[case_idx][0][0][0, 0].detach().cpu().numpy()
        flair_slice = get_middle_slice(flair_vol)

        gt_vol = all_cases[case_idx][1][0].detach().cpu().numpy()
        gt_slice = get_middle_slice(gt_vol)

        pred_all4, dice_all4, var_all4 = viz_preds_all4[case_idx]
        pred_flair, dice_flair, var_flair = viz_preds_flair_only[case_idx]

        save_path = save_segmentation_figure(
            case_idx,
            flair_slice,
            gt_slice,
            pred_all4,
            pred_flair,
            var_all4,
            var_flair,
            dice_all4,
            dice_flair,
            viz_dir
        )

        print(f"     Saved -> {save_path}")

    print(f"\n  -- Saving all-combination segmentation visualizations ({VIZ_CASES} cases) --")

    for case_idx in range(VIZ_CASES):
        flair_vol = all_cases[case_idx][0][0][0, 0].detach().cpu().numpy()
        flair_slice = get_middle_slice(flair_vol)

        gt_vol = all_cases[case_idx][1][0].detach().cpu().numpy()
        gt_slice = get_middle_slice(gt_vol)

        save_path = save_all_combos_figure(
            case_idx,
            flair_slice,
            gt_slice,
            viz_preds_by_combo[case_idx],
            viz_dir
        )

        print(f"     Saved -> {save_path}")

    print(f"\n  Visualization outputs saved to: {viz_dir}/")
    print("  Files:")
    print("     case_XXX_segmentation.png           — original 4-panel segmentation figures")
    print("     case_XXX_all_combinations.png       — all 15 combination prediction figures")

    print("\n  Use uncertainty_eval.py for:")
    print("     evaluation_results.csv")
    print("     uncertainty_per_case.csv")
    print("     uncertainty_summary.csv")
    print("     uncertainty_calibration.png")
    print("     correlation_heatmap.png")
    print("     correlation_matrix.csv")
