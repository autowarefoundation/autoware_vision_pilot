# AutoDrive trainer

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.data_utils.load_data_auto_drive import CURV_SCALE


class AverageMeter:
    """Running average over an epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------
TRAIN_MODE_CURVATURE = "curvature"   # freeze backbone + dist/flag heads
TRAIN_MODE_JOINT     = "joint"       # train everything end-to-end


class AutoDriveTrainer:
    """
    Losses
    ------
    curvature : L1( tanh_pred, curvature_gt )              — always
    distance  : L1( relu_pred, d_norm_gt )                 — only when dist_mask=True (CIPO present)
    flag      : BCEWithLogitsLoss( logits, flag_gt,
                  pos_weight≈0.418 )                        — always (joint mode only)

    Training modes
    --------------
    CURVATURE mode  — backbone frozen; only curvature_head updated.
                      Useful to verify the head can learn with pretrained features.
    JOINT mode      — backbone + all heads trained end-to-end.

    TensorBoard
    -----------
    Loss/train_*            per-step losses   (every log_every steps)
    Loss/train_avg_*        epoch-averaged losses
    Loss/val_*              validation losses
    Metrics/flag_acc_%      CIPO classification accuracy
    Metrics/dist_mae_m      distance MAE in metres (CIPO frames only)
    Metrics/lr
    Metrics/grad_norm       gradient norm before clipping
    Hist/flag_logits        histogram of flag logit distribution
    Hist/d_pred             histogram of distance predictions
    Hist/curv_pred          histogram of curvature predictions
    Visualization/sample    annotated frame image
    """

    _CIPO_POS_WEIGHT = torch.tensor([0.295 / 0.705])  # ≈ 0.418

    def __init__(self, tensorboard_dir: str = "runs",
                 train_mode: str = TRAIN_MODE_JOINT,
                 autospeed_ckpt: str = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_mode = train_mode
        print(f"AutoDriveTrainer — device: {self.device}  mode: {train_mode}")

        self.model = AutoDrive().to(self.device)

        # Optionally load pretrained backbone from AutoSpeed
        if autospeed_ckpt:
            self.model.load_backbone_from_autospeed(autospeed_ckpt)

        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # LR: 1e-4 for joint (E2E), 3e-4 for curvature-only (head only)
        self.learning_rate = 3e-4 if train_mode == TRAIN_MODE_CURVATURE else 1e-4
        self._setup_optimizer()

        self._l1  = nn.L1Loss()
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=self._CIPO_POS_WEIGHT.to(self.device)
        )

        # Per-step state
        self.loss:           torch.Tensor | None = None
        self.loss_distance:  float = 0.0
        self.loss_curvature: float = 0.0
        self.loss_flag:      float = 0.0
        self._grad_norm:     float = 0.0

        # Epoch-level running averages
        self.avg_total     = AverageMeter()
        self.avg_distance  = AverageMeter()
        self.avg_curvature = AverageMeter()
        self.avg_flag      = AverageMeter()

        # Last-batch tensors for histogram logging
        self._flag_logits_cpu: torch.Tensor | None = None
        self._d_pred_cpu:      torch.Tensor | None = None
        self._curv_pred_cpu:   torch.Tensor | None = None

        # Visualization
        self._img_prev_vis:  torch.Tensor | None = None
        self._d_pred_val:    float = 0.0
        self._d_gt_val:      float = 0.0
        self._curv_pred_val: float = 0.0
        self._curv_gt_val:   float = 0.0
        self._flag_pred_val: float = 0.0
        self._flag_gt_val:   float = 0.0

    # ------------------------------------------------------------------
    # Optimizer / parameter groups
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        """Build optimizer over trainable parameters only."""
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable, lr=self.learning_rate, weight_decay=1e-5)

    def _apply_train_mode(self):
        """Freeze/unfreeze parameters according to train_mode."""
        if self.train_mode == TRAIN_MODE_CURVATURE:
            # Freeze backbone; keep full head trainable.
            # Only curvature loss is computed — distance and flag losses are 0.
            # This lets the head conv+FC layers adapt to the task while the
            # pretrained backbone features stay intact.
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)
            for p in self.model.head.parameters():
                p.requires_grad_(True)
            n_frozen   = sum(p.numel() for p in self.model.backbone.parameters())
            n_trainable = sum(p.numel() for p in self.model.head.parameters())
            print(f"  [mode=curvature] backbone FROZEN ({n_frozen:,} params); "
                  f"full head trainable ({n_trainable:,} params). "
                  f"Curvature loss only.")
        else:
            # Everything trainable end-to-end
            for p in self.model.parameters():
                p.requires_grad_(True)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"  [mode=joint] all {n_total:,} parameters trainable.")

        self._setup_optimizer()  # rebuild after changing requires_grad

    def freeze_backbone(self):
        for p in self.model.backbone.parameters():
            p.requires_grad_(False)
        self._setup_optimizer()
        print("  Backbone FROZEN.")

    def unfreeze_backbone(self):
        for p in self.model.backbone.parameters():
            p.requires_grad_(True)
        self._setup_optimizer()
        print("  Backbone UNFROZEN.")

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def reset_averages(self):
        self.avg_total.reset()
        self.avg_distance.reset()
        self.avg_curvature.reset()
        self.avg_flag.reset()

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def set_batch(self, batch: dict):
        self.img_prev     = batch["img_prev"].to(self.device)
        self.img_curr     = batch["img_curr"].to(self.device)
        self.d_norm_gt    = batch["d_norm"].unsqueeze(1).to(self.device)
        self.curvature_gt = batch["curvature"].unsqueeze(1).to(self.device)
        self.flag_gt      = batch["flag"].unsqueeze(1).to(self.device)
        self.dist_mask    = batch["dist_mask"].to(self.device)

        self._img_prev_vis = batch["img_prev"][0]
        self._d_gt_val     = batch["d_norm"][0].item()
        self._curv_gt_val  = batch["curvature"][0].item()
        self._flag_gt_val  = batch["flag"][0].item()

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def run_model(self):
        d_pred, curv_pred, flag_logits = self.model(self.img_prev, self.img_curr)

        # Curvature — always active
        loss_c = self._l1(curv_pred, self.curvature_gt)

        if self.train_mode == TRAIN_MODE_CURVATURE:
            # Curvature-only: distance and flag losses are zero
            loss_d = torch.tensor(0.0, device=self.device)
            loss_f = torch.tensor(0.0, device=self.device)
        else:
            # Distance — only when CIPO is detected with a valid distance
            if self.dist_mask.any():
                mask_idx = self.dist_mask.unsqueeze(1)
                loss_d   = self._l1(d_pred[mask_idx], self.d_norm_gt[mask_idx])
            else:
                loss_d = torch.tensor(0.0, device=self.device)
            # Flag — always in joint mode
            loss_f = self._bce(flag_logits, self.flag_gt)

        self.loss           = loss_c + loss_d + loss_f
        self.loss_curvature = loss_c.item()
        self.loss_distance  = loss_d.item()
        self.loss_flag      = loss_f.item()

        n = self.img_prev.size(0)
        self.avg_total.update(self.loss.item(), n)
        self.avg_curvature.update(self.loss_curvature, n)
        self.avg_distance.update(self.loss_distance, n)
        self.avg_flag.update(self.loss_flag, n)

        # Store tensors for histogram logging (detach, move to CPU)
        with torch.no_grad():
            self._flag_logits_cpu = flag_logits.detach().cpu()
            self._d_pred_cpu      = d_pred.detach().cpu()
            self._curv_pred_cpu   = curv_pred.detach().cpu()

        self._d_pred_val    = d_pred[0].item()
        self._curv_pred_val = curv_pred[0].item()
        self._flag_pred_val = torch.sigmoid(flag_logits[0]).item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, batch: dict) -> tuple:
        """Returns (total, dist, curv, flag, flag_acc_pct, dist_mae_m)."""
        self.set_batch(batch)

        d_pred, curv_pred, flag_logits = self.model(self.img_prev, self.img_curr)

        loss_c = self._l1(curv_pred, self.curvature_gt)

        if self.dist_mask.any():
            mask_idx   = self.dist_mask.unsqueeze(1)
            loss_d     = self._l1(d_pred[mask_idx], self.d_norm_gt[mask_idx])
            dist_mae_m = (150.0 * (d_pred[mask_idx] - self.d_norm_gt[mask_idx]).abs()).mean().item()
        else:
            loss_d     = torch.tensor(0.0, device=self.device)
            dist_mae_m = 0.0

        loss_f = self._bce(flag_logits, self.flag_gt)
        total  = loss_c + loss_d + loss_f

        pred_label = (flag_logits > 0.0).float()
        flag_acc   = (pred_label == self.flag_gt).float().mean().item() * 100.0

        return (total.item(), loss_d.item(), loss_c.item(), loss_f.item(),
                flag_acc, dist_mae_m)

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def loss_backward(self):
        self.loss.backward()

    def run_optimizer(self):
        # Compute gradient norm before clipping (diagnostic)
        all_params = [p for p in self.model.parameters() if p.grad is not None]
        if all_params:
            self._grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=10.0
            ).item()
        else:
            self._grad_norm = 0.0

        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_loss(self) -> float:
        return self.loss.item()

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int, global_step: int, best_val_loss: float):
        print(f"Saving checkpoint → {path}")
        torch.save({
            "epoch":         epoch,
            "global_step":   global_step,
            "best_val_loss": best_val_loss,
            "train_mode":    self.train_mode,
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, int, float]:
        """
        Load checkpoint. Returns (start_epoch, global_step, best_val_loss).
        Handles both full-state and weights-only formats.
        """
        print(f"Loading checkpoint ← {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("  Optimizer state not restored (parameter groups changed).")
            start_epoch   = ckpt.get("epoch", 0)
            global_step   = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            saved_mode    = ckpt.get("train_mode", "unknown")
            print(f"  Resuming epoch {start_epoch + 1}, step {global_step}, "
                  f"best_val {best_val_loss:.4f}  (saved mode: {saved_mode})")
        else:
            self.model.load_state_dict(ckpt)
            start_epoch = global_step = 0
            best_val_loss = float("inf")
            print("  Weights-only checkpoint — counters reset.")

        return start_epoch, global_step, best_val_loss

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    # TensorBoard — per step
    # ------------------------------------------------------------------

    def log_train_step(self, step: int):
        """Per-step raw losses + diagnostic scalars."""
        self.writer.add_scalar("Loss/train_total",     self.get_loss(),       step)
        self.writer.add_scalar("Loss/train_curvature", self.loss_curvature,   step)
        self.writer.add_scalar("Loss/train_distance",  self.loss_distance,    step)
        self.writer.add_scalar("Loss/train_flag",      self.loss_flag,        step)
        self.writer.add_scalar("Metrics/grad_norm",    self._grad_norm,       step)

    def log_histograms(self, step: int):
        """Output distribution histograms — call every N steps."""
        if self._flag_logits_cpu is not None:
            self.writer.add_histogram("Hist/flag_logits", self._flag_logits_cpu, step)
            self.writer.add_histogram("Hist/flag_prob",
                                      torch.sigmoid(self._flag_logits_cpu), step)
        if self._d_pred_cpu is not None:
            self.writer.add_histogram("Hist/d_pred_norm", self._d_pred_cpu, step)
            # Convert to metres for readability
            self.writer.add_histogram("Hist/d_pred_m",
                                      150.0 * (1.0 - self._d_pred_cpu.clamp(0, 1)), step)
        if self._curv_pred_cpu is not None:
            self.writer.add_histogram("Hist/curv_pred", self._curv_pred_cpu, step)

    # ------------------------------------------------------------------
    # TensorBoard — per epoch
    # ------------------------------------------------------------------

    def log_train_epoch(self, epoch: int):
        self.writer.add_scalar("Loss/train_avg_total",     self.avg_total.avg,     epoch)
        self.writer.add_scalar("Loss/train_avg_curvature", self.avg_curvature.avg, epoch)
        self.writer.add_scalar("Loss/train_avg_distance",  self.avg_distance.avg,  epoch)
        self.writer.add_scalar("Loss/train_avg_flag",      self.avg_flag.avg,      epoch)
        self.writer.add_scalar("Metrics/lr",               self.learning_rate,     epoch)
        self.writer.flush()

    def log_val_epoch(self, total: float, dist: float, curv: float, flag: float,
                      flag_acc: float, dist_mae_m: float, epoch: int):
        self.writer.add_scalar("Loss/val_total",      total,      epoch)
        self.writer.add_scalar("Loss/val_curvature",  curv,       epoch)
        self.writer.add_scalar("Loss/val_distance",   dist,       epoch)
        self.writer.add_scalar("Loss/val_flag",       flag,       epoch)
        self.writer.add_scalar("Metrics/flag_acc_%",  flag_acc,   epoch)
        self.writer.add_scalar("Metrics/dist_mae_m",  dist_mae_m, epoch)
        self.writer.flush()

    def log_test(self, total: float, dist: float, curv: float, flag: float,
                 flag_acc: float, dist_mae_m: float):
        self.writer.add_scalar("Loss/test_total",         total,      0)
        self.writer.add_scalar("Loss/test_curvature",     curv,       0)
        self.writer.add_scalar("Loss/test_distance",      dist,       0)
        self.writer.add_scalar("Loss/test_flag",          flag,       0)
        self.writer.add_scalar("Metrics/test_flag_acc_%", flag_acc,   0)
        self.writer.add_scalar("Metrics/test_dist_mae_m", dist_mae_m, 0)
        self.writer.flush()

    def log_train_loss(self, step: int):
        self.log_train_step(step)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def save_visualization(self, step: int):
        """Write an annotated sample image to TensorBoard."""
        if self._img_prev_vis is None:
            return

        d_pred_m = 150.0 * (1.0 - self._d_pred_val)
        d_gt_m   = 150.0 * (1.0 - self._d_gt_val)

        img_np = self._img_prev_vis.permute(1, 2, 0).numpy()
        img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                                 gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor("#1e1e1e")

        # Left: camera image with overlay
        ax_img = axes[0]
        ax_img.imshow(img_np)
        ax_img.set_title(f"Step {step}", color="white", fontsize=9)
        ax_img.axis("off")

        # Overlay text — curvature displayed in physical units (1/m)
        curv_pred_m = self._curv_pred_val * CURV_SCALE
        curv_gt_m   = self._curv_gt_val   * CURV_SCALE
        txt_lines = [
            f"dist pred : {d_pred_m:6.1f} m   GT: {d_gt_m:6.1f} m",
            f"curv pred : {curv_pred_m:+.5f} 1/m  GT: {curv_gt_m:+.5f} 1/m",
            f"flag prob : {self._flag_pred_val:.3f}        GT: {int(self._flag_gt_val)}",
        ]
        ax_img.text(
            10, 30, "\n".join(txt_lines),
            color="lime", fontsize=7, family="monospace",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=4),
        )

        # Right: bar chart of three predictions vs GT
        # Curvature is already normalised ∈ [-1, 1] (GT÷CURV_SCALE, pred=Tanh).
        ax_bar = axes[1]
        ax_bar.set_facecolor("#2a2a2a")
        labels  = ["d_norm", "curv (norm)", "flag_prob"]
        pred_vals = [
            float(np.clip(self._d_pred_val,    0, 1)),
            float(np.clip(self._curv_pred_val, -1, 1)),
            float(self._flag_pred_val),
        ]
        gt_vals = [
            float(np.clip(self._d_gt_val,    0, 1)),
            float(np.clip(self._curv_gt_val, -1, 1)),
            float(self._flag_gt_val),
        ]
        x = np.arange(len(labels))
        w = 0.35
        ax_bar.bar(x - w/2, pred_vals, w, label="pred", color="#4fc3f7", alpha=0.85)
        ax_bar.bar(x + w/2, gt_vals,   w, label="GT",   color="#ef9a9a", alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, color="white", fontsize=7)
        ax_bar.set_ylim(-1.1, 1.1)
        ax_bar.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax_bar.legend(fontsize=7, labelcolor="white",
                      facecolor="#333333", edgecolor="none")
        ax_bar.tick_params(colors="white")
        ax_bar.set_title("Pred vs GT", color="white", fontsize=9)

        plt.tight_layout()
        self.writer.add_figure("Visualization/sample", fig, global_step=step)
        plt.close(fig)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("AutoDriveTrainer: finished.")
