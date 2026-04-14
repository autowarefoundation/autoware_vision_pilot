"""
Main training loop for AutoDrive (ZOD dataset).

All outputs live under {zod_root}/training/autodrive/{run_name}/:
    checkpoints/AutoDrive_last.pth   — saved every epoch (full state)
    checkpoints/AutoDrive_best.pth   — saved when val loss improves
    tensorboard/                     — TensorBoard event files

─────────────────────────────────────────────────────────
Usage — Phase 1: curvature head only (with pretrained backbone)
─────────────────────────────────────────────────────────
  python Models/training/train_auto_drive.py \\
      --root ~/Downloads/data/zod \\
      --autospeed-ckpt ~/Downloads/data/zod/models/autospeed.pt \\
      --train-mode curvature \\
      --epochs 20 \\
      --batch-size 16 --workers 2

─────────────────────────────────────────────────────────
Usage — Phase 2: joint training of all heads (resume from Phase 1)
─────────────────────────────────────────────────────────
  python Models/training/train_auto_drive.py \\
      --root ~/Downloads/data/zod \\
      --autospeed-ckpt ~/Downloads/data/zod/models/autospeed.pt \\
      --train-mode joint \\
      --checkpoint <phase1_best.pth> \\
      --epochs 50 \\
      --batch-size 16 --workers 2

─────────────────────────────────────────────────────────
Resume after crash
─────────────────────────────────────────────────────────
  Add --resume  (auto-loads AutoDrive_last.pth from the run dir)
  or  --checkpoint <explicit path>

TensorBoard:
  tensorboard --logdir ~/Downloads/data/zod/training/autodrive/<run_name>/tensorboard/

Scalars
-------
  Loss/train_avg_*         epoch-averaged train losses (smooth)
  Loss/train_*             per-step losses (noisy, every --log-every steps)
  Loss/val_*               validation losses
  Metrics/flag_acc_%       CIPO flag accuracy
  Metrics/dist_mae_m       distance MAE in metres
  Metrics/grad_norm        gradient norm (for diagnosing exploding gradients)
  Hist/flag_logits         flag logit distribution (per --log-every steps)
  Hist/d_pred_*            distance pred distribution
  Hist/curv_pred           curvature pred distribution
  Visualization/sample     annotated image (per --vis-every steps)
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
from Models.training.auto_drive_trainer import (
    AutoDriveTrainer,
    TRAIN_MODE_CURVATURE,
    TRAIN_MODE_JOINT,
)


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _run_val(trainer: AutoDriveTrainer, loader: DataLoader):
    """Average validation metrics across all batches."""
    total = dist = curv = flag = acc = mae = steer_mae = 0.0
    n = 0
    for batch in loader:
        t, d, c, f, a, m, s = trainer.validate(batch)
        total += t; dist += d; curv += c; flag += f; acc += a; mae += m; steer_mae += s
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return total/n, dist/n, curv/n, flag/n, acc/n, mae/n, steer_mae/n


def main():
    parser = ArgumentParser()
    parser.add_argument("--root",         required=True,
                        help="ZOD dataset root — outputs go under {root}/training/autodrive/<run-name>/")
    parser.add_argument("--run-name",     default="",
                        help="Sub-folder name for this run (default: auto-numbered run001, run002, …)")
    parser.add_argument("--train-mode",   default=TRAIN_MODE_JOINT,
                        choices=[TRAIN_MODE_CURVATURE, TRAIN_MODE_JOINT],
                        help="'curvature' — train curvature head only (backbone frozen); "
                             "'joint' — train all heads end-to-end")
    parser.add_argument("--autospeed-ckpt", default="",
                        help="Path to autospeed.pt — initialises backbone from pretrained weights")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from AutoDrive_last.pth in the run dir")
    parser.add_argument("--checkpoint",   default="",
                        help="Explicit checkpoint path (overrides --resume)")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--log-every",    type=int, default=100,
                        help="Log per-step scalars + histograms every N steps")
    parser.add_argument("--vis-every",    type=int, default=500,
                        help="Save visualization image to TensorBoard every N steps")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    base_dir = Path(args.root) / "training" / "autodrive"

    if args.run_name:
        run_name = args.run_name
    else:
        # Auto-number: run001, run002, …
        existing = sorted(base_dir.glob("run[0-9][0-9][0-9]"))
        run_name = f"run{len(existing) + 1:03d}"

    run_dir  = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    tb_dir   = run_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    ckpt_last = str(ckpt_dir / "AutoDrive_last.pth")
    ckpt_best = str(ckpt_dir / "AutoDrive_best.pth")

    print(f"Run         : {run_dir}")
    print(f"Mode        : {args.train_mode}")
    print(f"Checkpoints : {ckpt_dir}")
    print(f"TensorBoard : {tb_dir}")
    print(f"  tensorboard --logdir {tb_dir}")
    if args.autospeed_ckpt:
        print(f"AutoSpeed   : {args.autospeed_ckpt}")

    # ------------------------------------------------------------------
    # Dataset + DataLoaders
    # ------------------------------------------------------------------
    data = LoadDataAutoDrive(args.root)

    train_loader = DataLoader(
        data.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )
    test_loader = DataLoader(
        data.test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )

    steps_per_epoch = len(train_loader)
    print(f"Train: {len(data.train):,}  Val: {len(data.val):,}  Test: {len(data.test):,} pairs")
    print(f"Steps per epoch : {steps_per_epoch:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = AutoDriveTrainer(
        tensorboard_dir=str(tb_dir),
        train_mode=args.train_mode,
        autospeed_ckpt=args.autospeed_ckpt,
    )
    trainer._apply_train_mode()
    trainer.zero_grad()

    # Resume state
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    resume_path = ""
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            resume_path = args.checkpoint
        else:
            print(f"  WARNING: --checkpoint not found: {args.checkpoint}")
    elif args.resume:
        if Path(ckpt_last).exists():
            resume_path = ckpt_last
        else:
            print("  --resume: no AutoDrive_last.pth found, starting fresh.")

    if resume_path:
        start_epoch, global_step, best_val_loss = trainer.load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # LR schedule helper
    # ------------------------------------------------------------------
    def _get_lr(epoch: int, mode: str) -> float:
        """
        Curvature mode (head-only):
            epochs 0-9  : 3e-4
            epochs 10+  : 3e-5
        Joint mode (E2E):
            epochs 0-14 : 1e-4
            epochs 15-34: 3e-5
            epochs 35+  : 1e-5
        """
        if mode == TRAIN_MODE_CURVATURE:
            return 3e-4 if epoch < 10 else 3e-5
        else:
            if epoch < 15:  return 1e-4
            if epoch < 35:  return 3e-5
            return 1e-5

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"(mode={args.train_mode}, global_step={global_step})")

        lr = _get_lr(epoch, args.train_mode)
        trainer.set_learning_rate(lr)

        trainer.set_train_mode()
        trainer.reset_averages()

        p_bar = tqdm.tqdm(
            train_loader, total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}",
        )

        for batch in p_bar:
            trainer.set_batch(batch)
            trainer.run_model()
            trainer.loss_backward()
            trainer.run_optimizer()

            global_step += 1

            p_bar.set_description(
                f"[{epoch+1}/{args.epochs}]  "
                f"loss {trainer.avg_total.avg:.4f}  "
                f"c {trainer.avg_curvature.avg:.4f}  "
                f"d {trainer.avg_distance.avg:.4f}  "
                f"f {trainer.avg_flag.avg:.4f}  "
                f"|g| {trainer._grad_norm:.2f}"
            )

            if global_step % args.log_every == 0:
                trainer.log_train_step(global_step)
                trainer.log_histograms(global_step)

            if global_step % args.vis_every == 0:
                trainer.save_visualization(global_step)

        trainer.log_train_epoch(epoch + 1)

        # Save checkpoint every epoch
        trainer.save_checkpoint(ckpt_last, epoch + 1, global_step, best_val_loss)
        ckpt_epoch = str(ckpt_dir / f"AutoDrive_epoch{epoch+1:03d}.pth")
        trainer.save_checkpoint(ckpt_epoch, epoch + 1, global_step, best_val_loss)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        print("  Validating...")
        trainer.set_eval_mode()
        with torch.no_grad():
            v_total, v_dist, v_curv, v_flag, v_acc, v_mae, v_steer_mae = _run_val(trainer, val_loader)

        trainer.log_val_epoch(v_total, v_dist, v_curv, v_flag, v_acc, v_mae, v_steer_mae, epoch + 1)
        trainer.save_visualization(epoch + 1, split="val")
        print(
            f"  [Val] loss {v_total:.4f}  "
            f"steer_mae {v_steer_mae:.2f}°  d {v_dist:.4f}  f {v_flag:.4f} c {v_curv:.4f}"
            f"flag_acc {v_acc:.1f}%  dist_mae {v_mae:.1f} m"
        )

        if v_total < best_val_loss:
            best_val_loss = v_total
            trainer.save_checkpoint(ckpt_best, epoch + 1, global_step, best_val_loss)
            print(f"  *** New best val loss: {best_val_loss:.4f} → AutoDrive_best.pth")

        trainer.set_train_mode()

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\nTest set evaluation...")
    trainer.set_eval_mode()
    with torch.no_grad():
        t_total, t_dist, t_curv, t_flag, t_acc, t_mae, t_steer_mae = _run_val(trainer, test_loader)
    trainer.log_test(t_total, t_dist, t_curv, t_flag, t_acc, t_mae, t_steer_mae)
    print(
        f"[Test] loss {t_total:.4f}  "
        f"steer_mae {t_steer_mae:.2f}°  d {t_dist:.4f}  f {t_flag:.4f} c {t_curv:.4f}"  
        f"flag_acc {t_acc:.1f}%  dist_mae {t_mae:.1f} m"
    )

    trainer.cleanup()


if __name__ == "__main__":
    main()
