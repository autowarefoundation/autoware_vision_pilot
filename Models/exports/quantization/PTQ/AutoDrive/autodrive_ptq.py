#!/usr/bin/env python3
"""
PT2E Post-Training Static Quantization (PTQ)

Pipeline
────────
  float model
      │
  torch.export.export        ← captures ATen graph
      │
  prepare_pt2e               ← inserts MinMax observers
      │
  calibrate                  ← run representative data, collect stats
      │
  convert_pt2e               ← replace fake-quant with Q/DQ ops → INT8 model
      │
  validate                   ← check accuracy before deploying
      │
  torch.onnx.export          ← INT8 ONNX for deployment

OOP design: subclass PT2EQuantizerBase and implement the 6 abstract methods.
AutoDrivePTQ shows exactly how to do it.
To quantize AutoSpeed, create AutoSpeedPTQ the same way.

Usage
─────
  python Models/exports/quantization/PTQ/AutoDrive/autodrive_ptq.py \\
      --root       ~/data/zod \\
      --checkpoint ~/data/zod/training/autodrive/run002/checkpoints/AutoDrive_best.pth \\
      --out-dir    ~/data/zod/training/autodrive_ptq/ptq001 \\
      --calib-samples 500 \\
      --val-fraction 0.10

  # skip ONNX export:
  python Models/exports/quantization/PTQ/AutoDrive/autodrive_ptq.py ... --no-onnx
"""

from __future__ import annotations

import copy
import itertools
import math
import sys
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
import torchao.quantization.pt2e as pt2e_utils
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


# ══════════════════════════════════════════════════════════════════════════════
#  Abstract base — swap model + data by subclassing
# ══════════════════════════════════════════════════════════════════════════════

class PT2EQuantizerBase(ABC):
    """
    Generic PT2E PTQ pipeline.

    Override the 6 abstract methods below to quantize any model.
    The run() method drives the full pipeline — never touch it.
    """

    def __init__(self, args) -> None:
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 6 abstract methods to implement ──────────────────────────────────────

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Return a fresh eval-mode float model on CPU."""

    @abstractmethod
    def load_float_weights(self, model: nn.Module, path: str) -> None:
        """Load checkpoint weights into model in-place."""

    @abstractmethod
    def build_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return (calib_loader, int8_val_loader, fp32_val_loader).

        calib_loader    — batch_size == args.batch_size, drop_last=True
                          (must match the static export batch size)
        int8_val_loader — same batch_size, drop_last=True
                          (used for INT8 PT2E PyTorch validation)
        fp32_val_loader — same batch_size, drop_last=True
                          (used for FP32 and ONNX validation)
        """

    @abstractmethod
    def make_example_inputs(self) -> tuple:
        """Return CPU example tensors with batch_size == args.batch_size."""

    @abstractmethod
    def run_model(self, model: Any, batch: Any) -> tuple:
        """Run one batch, return tuple of output tensors."""

    @abstractmethod
    def compute_metrics(self, outputs: tuple, batch: Any) -> dict[str, float]:
        """Return per-batch metric dict from model outputs + GT."""

    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Optional subclass hook to make export/prepare numerically stable."""
        return model

    # ── Pipeline steps (shared, no need to touch these) ──────────────────────

    def _step1_export_and_prepare(self, float_model: nn.Module) -> Any:
        print("  [1/4] torch.export + prepare_pt2e …")
        m  = self.prepare_model_for_export(copy.deepcopy(float_model).eval().cpu())
        ex = self.make_example_inputs()   # batch size == args.batch_size

        # Static export — the guard fixes batch == args.batch_size.
        # Calibration and benchmark loaders use the same batch size with
        # drop_last=True so the PT2E/ONNX graphs see the shape they exported.
        exported = torch.export.export(m, ex, strict=False).module()

        quantizer = self._make_quantizer()
        prepared = prepare_pt2e(exported, quantizer)
        return prepared

    def _make_quantizer(self) -> XNNPACKQuantizer:
        # Force per-tensor PTQ:
        #   activations: int8 per-tensor affine
        #   weights:     int8 per-tensor symmetric
        #
        # XNNPACK/PT2E does not expose int8 bias here; bias is handled by the
        # backend/reference op, typically as fp32 or int32 accumulator data.
        qconfig = get_symmetric_quantization_config(
            is_per_channel=False,
            is_qat=False,
        )
        print("  Quantization config: per-tensor int8 activations + per-tensor int8 weights.")
        return XNNPACKQuantizer().set_global(qconfig)

    def _step2_calibrate(self, prepared: Any, calib_loader: DataLoader) -> None:
        print(f"  [2/4] Calibrating ({self.args.calib_samples} samples) on {self.device} …")
        # Move to GPU for fast calibration; observers live in the graph buffers
        prepared.to(self.device)
        pt2e_utils.move_exported_model_to_eval(prepared)
        collected = 0
        target_batches = min(
            len(calib_loader),
            max(1, math.ceil(self.args.calib_samples / self.args.batch_size)),
        )
        iterator = itertools.islice(calib_loader, target_batches)
        with torch.no_grad():
            for batch in tqdm.tqdm(iterator, total=target_batches, desc="  calib", leave=False):
                self.run_model(prepared, batch)
                collected += self._batch_size(batch)
                if collected >= self.args.calib_samples:
                    break
        prepared.apply(pt2e_utils.disable_observer)
        print(f"  Calibrated on {collected} samples — observers frozen.")

    def _step3_convert(self, prepared: Any) -> Any:
        print("  [3/4] convert_pt2e → INT8 model …")
        # convert_pt2e requires CPU; move back to device after conversion
        int8 = convert_pt2e(copy.deepcopy(prepared).cpu())
        pt2e_utils.move_exported_model_to_eval(int8)
        int8.to(self.device)
        return int8

    def _step4_export_onnx(self, int8_model: Any, out_path: Path) -> None:
        print(f"  [4/4] Exporting INT8 ONNX → {out_path.name} …")
        ex           = self.make_example_inputs()
        input_names  = self._onnx_input_names(len(ex))
        output_names = self._onnx_output_names()
        model_cpu    = int8_model.cpu()
        pt2e_utils.move_exported_model_to_eval(model_cpu)
        pt2e_utils.allow_exported_model_train_eval(model_cpu)
        model_cpu.eval()
        for module in model_cpu.modules():
            module.training = False
        torch.onnx.export(
            model_cpu, ex, str(out_path),
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            external_data=False,
        )
        print(f"  ONNX saved ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, model: Any, val_loader: DataLoader, label: str = "") -> dict[str, float]:
        agg: dict[str, float] = {}
        n = 0
        timed_ms = 0.0
        timed_samples = 0
        max_b = self._validation_batch_limit(val_loader)
        total = len(val_loader) if max_b <= 0 else min(len(val_loader), max_b)
        iterator = val_loader if max_b <= 0 else itertools.islice(val_loader, max_b)
        for batch in tqdm.tqdm(iterator, total=total, desc=f"  val [{label}]", leave=False):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.run_model(model, batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            timed_ms += (time.perf_counter() - t0) * 1000.0
            timed_samples += self._batch_size(batch)
            for k, v in self.compute_metrics(out, batch).items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        metrics = {k: v / max(n, 1) for k, v in agg.items()}
        metrics["ms_per_sample"] = timed_ms / max(timed_samples, 1)
        return metrics

    def _validation_batch_limit(self, val_loader: DataLoader) -> int:
        """Return 0 for full validation, otherwise the max number of batches."""
        if self.args.val_batches > 0:
            return min(len(val_loader), self.args.val_batches)
        frac = float(getattr(self.args, "val_fraction", 1.0))
        if frac <= 0.0 or frac >= 1.0:
            return 0
        return max(1, min(len(val_loader), math.ceil(len(val_loader) * frac)))

    def _print_validation_budget(self, val_loader: DataLoader) -> None:
        total_batches = len(val_loader)
        max_b = self._validation_batch_limit(val_loader)
        use_batches = total_batches if max_b <= 0 else max_b
        approx_samples = use_batches * self.args.batch_size
        val_fraction = float(getattr(self.args, "val_fraction", 1.0))
        if self.args.val_batches > 0:
            why = f"--val-batches={self.args.val_batches}"
        elif 0.0 < val_fraction < 1.0:
            why = f"{val_fraction:.0%} of validation"
        else:
            why = "full validation"
        print(f"  Benchmark budget: {use_batches:,}/{total_batches:,} batches "
              f"(~{approx_samples:,} samples, {why}; drop_last=True)")

    # ── Comparison table ──────────────────────────────────────────────────────

    def _print_table(self, cols: dict[str, dict]) -> None:
        """cols = OrderedDict of  label → metrics_dict."""
        W   = 14
        n   = len(cols)
        sep = "─" * (28 + W * n)
        header = "  vs  ".join(cols.keys())
        print(f"\n  ┌{sep}┐")
        print(f"  │  {header:<{26 + W * n}}│")
        print(f"  ├{sep}┤")
        hdr_row = f"  │  {'Metric':<26}" + "".join(f"{label:>{W}}" for label in cols) + "  │"
        print(hdr_row)
        print(f"  ├{sep}┤")
        keys = list(next(iter(cols.values())).keys())
        for k in keys:
            row = f"  │  {k:<26}"
            for m in cols.values():
                v = m.get(k, float("nan"))
                row += f"{v:>{W}.4f}" if not (v != v) else f"{'N/A':>{W}}"
            print(row + "  │")
        print(f"  └{sep}┘\n")

    # ── High-level entry point ────────────────────────────────────────────────

    def run(self) -> None:
        args    = self.args
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n  PT2E PTQ  |  {self.__class__.__name__}  |  {self.device}\n{'='*60}\n")

        # ── 1. load float model ───────────────────────────────────────────────
        float_model = self.build_model()
        self.load_float_weights(float_model, args.checkpoint)
        float_model.eval().cpu()

        # ── 2. data ───────────────────────────────────────────────────────────
        calib_loader, int8_val_loader, fp32_val_loader = self.build_data()

        # ── 3. full quantization pipeline ────────────────────────────────────
        #      export (CPU) → prepare → calibrate (GPU) → convert → ONNX
        prepared   = self._step1_export_and_prepare(float_model)
        self._step2_calibrate(prepared, calib_loader)
        int8_model = self._step3_convert(prepared)    # lands on self.device

        ckpt = out_dir / "model_int8_pt2e.pth"
        torch.save(int8_model.state_dict(), ckpt)
        print(f"  Saved INT8 PT2E checkpoint → {ckpt.name}")

        onnx_path: Path | None = None
        if not args.no_onnx:
            onnx_path = out_dir / "model_int8_pt2e.onnx"
            self._step4_export_onnx(int8_model, onnx_path)

        # ── 4. benchmark all models together ─────────────────────────────────
        print(f"\n{'─'*60}")
        print("  Benchmarking FP32 vs INT8 PT2E" +
              (" vs INT8 ONNX" if onnx_path else "") + " …\n")
        self._print_validation_budget(fp32_val_loader)

        fp32_model = copy.deepcopy(float_model).to(self.device).eval()
        fp32_metrics  = self._validate(fp32_model,   fp32_val_loader, label="FP32")

        int8_model.to(self.device)
        int8_metrics  = self._validate(int8_model, int8_val_loader, label="INT8 PT2E")

        onnx_metrics: dict | None = None
        if onnx_path:
            onnx_metrics = self._validate_onnx(str(onnx_path), fp32_val_loader, label="INT8 ONNX")

        cols: dict = {"FP32": fp32_metrics, "INT8 PT2E": int8_metrics}
        if onnx_metrics is not None:
            cols["INT8 ONNX"] = onnx_metrics
        self._print_table(cols)

        print(f"  Outputs → {out_dir}\n  Done.\n")

    # ── ONNX validation ───────────────────────────────────────────────────────

    def _create_ort_session(self, onnx_path: str):
        import onnxruntime as ort

        mode = getattr(self.args, "ort_provider", "auto")
        available = ort.get_available_providers()
        wants_cuda = mode in {"auto", "cuda"}

        if mode == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif mode == "cuda":
            raise RuntimeError(
                "ONNX Runtime CUDAExecutionProvider is not available. "
                f"Available providers: {available}"
            )
        else:
            print(f"  ORT CUDA provider unavailable; using CPU. Available providers: {available}")
            providers = ["CPUExecutionProvider"]

        try:
            sess = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as exc:
            if wants_cuda and mode == "auto" and providers[0] == "CUDAExecutionProvider":
                print("  ORT CUDA session creation failed; retrying with CPUExecutionProvider.")
                print("  Start Python with LD_LIBRARY_PATH containing CUDA 12/cuDNN 9 libs to use ORT CUDA.")
                print(f"  ORT CUDA error: {exc}")
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            else:
                raise

        active = sess.get_providers()
        if wants_cuda and "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            msg = (
                "ORT requested CUDAExecutionProvider but the active providers are "
                f"{active}. Check LD_LIBRARY_PATH for CUDA 12/cuDNN 9 dependencies."
            )
            if mode == "cuda":
                raise RuntimeError(msg)
            print(f"  {msg}")

        print(f"  ORT providers: {', '.join(active)}")
        return sess

    @staticmethod
    def _onnx_static_batch(sess: Any) -> int | None:
        batch_dims = [inp.shape[0] for inp in sess.get_inputs()]
        if batch_dims and all(isinstance(d, int) for d in batch_dims) and len(set(batch_dims)) == 1:
            return int(batch_dims[0])
        return None

    @torch.no_grad()
    def _validate_onnx(
        self, onnx_path: str, val_loader: DataLoader, label: str = ""
    ) -> dict[str, float]:
        import numpy as np

        sess  = self._create_ort_session(onnx_path)
        names = [inp.name for inp in sess.get_inputs()]
        static_batch = self._onnx_static_batch(sess)

        agg: dict[str, float] = {}
        n   = 0
        timed_ms = 0.0
        timed_samples = 0
        max_b = self._validation_batch_limit(val_loader)
        total = len(val_loader) if max_b <= 0 else min(len(val_loader), max_b)
        iterator = val_loader if max_b <= 0 else itertools.islice(val_loader, max_b)
        for batch in tqdm.tqdm(iterator, total=total, desc=f"  val [{label}]", leave=False):
            actual_bs = self._batch_size(batch)
            feed = {}
            for name, t in zip(names, self._batch_inputs(batch)):
                arr = t.cpu().numpy().astype(np.float32)
                if static_batch is not None and arr.shape[0] != static_batch:
                    if arr.shape[0] > static_batch:
                        raise ValueError(
                            f"ONNX model expects batch {static_batch}, got {arr.shape[0]}. "
                            "Use a matching --batch-size or export again."
                        )
                    pad_width = [(0, 0)] * arr.ndim
                    pad_width[0] = (0, static_batch - arr.shape[0])
                    arr = np.pad(arr, pad_width, mode="constant")
                feed[name] = arr
            t0 = time.perf_counter()
            raw = sess.run(None, feed)
            timed_ms += (time.perf_counter() - t0) * 1000.0
            timed_samples += actual_bs
            outputs = tuple(torch.from_numpy(r[:actual_bs]) for r in raw)
            for k, v in self.compute_metrics(outputs, batch).items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        metrics = {k: v / max(n, 1) for k, v in agg.items()}
        metrics["ms_per_sample"] = timed_ms / max(timed_samples, 1)
        return metrics

    # ── helpers subclasses may override ──────────────────────────────────────

    def _onnx_input_names(self, n: int) -> list[str]:
        return [f"input_{i}" for i in range(n)]

    def _onnx_output_names(self) -> list[str]:
        return ["output"]

    def _batch_size(self, batch: Any) -> int:
        tensors = self._batch_inputs(batch)
        return tensors[0].size(0)

    def _batch_inputs(self, batch: Any) -> list[torch.Tensor]:
        """Extract input tensors from a batch. Override if needed."""
        if isinstance(batch, (list, tuple)):
            return list(batch[:len(self.make_example_inputs())])
        raise NotImplementedError("Override _batch_inputs for your data format.")


# ══════════════════════════════════════════════════════════════════════════════
#  AutoDrive — concrete implementation
# ══════════════════════════════════════════════════════════════════════════════

class AutoDrivePTQ(PT2EQuantizerBase):
    """PT2E PTQ for the AutoDrive E2E model."""

    _POS_W   = torch.tensor([0.295 / 0.705])
    _CW, _DW = 2.0, 2.0
    _THR     = 0.65
    _WB      = 2.984
    _SCR     = 16.8

    def build_model(self) -> nn.Module:
        from Models.model_components.autodrive.autodrive_network import AutoDrive
        return AutoDrive().eval().cpu()

    def load_float_weights(self, model: nn.Module, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd)
        print(f"  Loaded weights: {Path(path).name}")

    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        if self.args.no_fuse_conv_bn:
            return model

        from Models.model_components.common_layers import Conv

        # AutoDrive calls the same backbone twice.  PT2E prepare folds BatchNorm
        # patterns in the exported graph; pre-fusing avoids folding the shared
        # Conv+BN weights more than once.
        def _fuse_conv_bn(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> nn.Conv2d:
            fused = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
                padding_mode=conv.padding_mode,
            ).requires_grad_(False).to(conv.weight.device)

            w_conv = conv.weight.detach().clone().view(conv.out_channels, -1)
            scale = norm.weight.detach() / torch.sqrt(norm.running_var.detach() + norm.eps)
            fused.weight.copy_((w_conv * scale.reshape(-1, 1)).view_as(fused.weight))

            b_conv = (
                torch.zeros(conv.out_channels, device=conv.weight.device)
                if conv.bias is None
                else conv.bias.detach()
            )
            b_norm = norm.bias.detach() - scale * norm.running_mean.detach()
            fused.bias.copy_(b_conv * scale + b_norm)
            return fused

        fused_count = 0
        with torch.no_grad():
            for module in model.modules():
                if type(module) is Conv and hasattr(module, "norm"):
                    module.conv = _fuse_conv_bn(module.conv, module.norm)
                    module.forward = module.fuse_forward
                    delattr(module, "norm")
                    fused_count += 1
        print(f"  Fused {fused_count} Conv+BatchNorm blocks before PT2E export.")
        return model

    def build_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
        args = self.args
        bs   = args.batch_size

        def _collate(b):
            return {k: torch.stack([x[k] for x in b]) for k in b[0]}

        data = LoadDataAutoDrive(args.root)
        # drop_last=True so every validation batch matches the static PT2E export guard.
        calib_loader    = DataLoader(data.val, bs, shuffle=True,  drop_last=True,
                                     num_workers=args.workers, collate_fn=_collate)
        int8_val_loader = DataLoader(data.val, bs, shuffle=False, drop_last=True,
                                     num_workers=args.workers, collate_fn=_collate)
        fp32_val_loader = DataLoader(data.val, bs, shuffle=False, drop_last=True,
                                     num_workers=args.workers, collate_fn=_collate)
        dropped = len(data.val) - len(fp32_val_loader) * bs
        print(f"  Val set: {len(data.val):,} samples  "
              f"({len(fp32_val_loader):,} full batches × {bs}, dropped tail={dropped})")
        return calib_loader, int8_val_loader, fp32_val_loader

    def make_example_inputs(self) -> tuple:
        bs = self.args.batch_size
        return (torch.randn(bs, 3, 512, 1024), torch.randn(bs, 3, 512, 1024))

    def run_model(self, model: Any, batch: Any) -> tuple:
        ip = batch["img_prev"].to(self.device)
        ic = batch["img_curr"].to(self.device)
        return model(ip, ic)

    def compute_metrics(self, outputs: tuple, batch: Any) -> dict[str, float]:
        from Models.data_utils.load_data_auto_drive import CURV_SCALE
        d_pred, c_pred, f_logit = (t.cpu() for t in outputs)

        d_gt  = batch["d_norm"].unsqueeze(1)
        c_gt  = batch["curvature"].unsqueeze(1)
        f_gt  = batch["flag"].unsqueeze(1).float()
        mask  = batch["dist_mask"]

        l1  = nn.L1Loss()
        bce = nn.BCEWithLogitsLoss(pos_weight=self._POS_W)
        lc  = l1(c_pred, c_gt).item()
        ld  = l1(d_pred[mask.unsqueeze(1)], d_gt[mask.unsqueeze(1)]).item() if mask.any() else 0.0
        dm  = (150.0 * (d_pred[mask.unsqueeze(1)] - d_gt[mask.unsqueeze(1)]).abs()).mean().item() if mask.any() else 0.0
        lf  = bce(f_logit, f_gt).item()
        total = self._CW * lc + self._DW * ld + lf

        facc  = ((torch.sigmoid(f_logit) > self._THR).float() == f_gt).float().mean().item() * 100.0
        ss    = self._SCR * (180.0 / math.pi)
        smae  = (torch.atan(c_pred * CURV_SCALE * self._WB) * ss
                 - torch.atan(c_gt * CURV_SCALE * self._WB) * ss).abs().mean().item()

        return dict(total=total, flag_acc=facc, steer_mae_deg=smae, dist_mae_m=dm)

    def _onnx_output_names(self) -> list[str]:
        return ["distance", "curvature", "flag_logit"]

    def _onnx_input_names(self, n: int) -> list[str]:
        return ["image_prev", "image_curr"]

    def _batch_inputs(self, batch: Any) -> list[torch.Tensor]:
        return [batch["img_prev"], batch["img_curr"]]

    def _batch_size(self, batch: Any) -> int:
        return batch["img_prev"].size(0)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = ArgumentParser(description="PT2E PTQ — AutoDrive")
    p.add_argument("--root",          required=True,  help="ZOD root directory")
    p.add_argument("--checkpoint",    required=True,  help="FP32 AutoDrive .pth")
    p.add_argument("--out-dir",       required=True,  help="Output directory")
    p.add_argument("--calib-samples", type=int, default=500,
                   help="Calibration samples (default 500)")
    p.add_argument("--val-batches",   type=int, default=0,
                   help="Override validation/benchmark limit in batches; 0 uses --val-fraction")
    p.add_argument("--val-fraction",  type=float, default=0.10,
                   help="Validation fraction for benchmarking when --val-batches=0; 0 or >=1 = full val set (default 0.10)")
    p.add_argument("--batch-size",    type=int, default=16)
    p.add_argument("--workers",       type=int, default=4)
    p.add_argument("--ort-provider",  choices=("auto", "cuda", "cpu"), default="auto",
                   help="ONNX Runtime provider for ONNX benchmark (default auto; cuda fails if CUDA EP cannot load)")
    p.add_argument("--no-fuse-conv-bn", action="store_true",
                   help="Disable pre-export Conv+BatchNorm fusion for debugging")
    p.add_argument("--no-onnx",       action="store_true",
                   help="Skip ONNX export")
    args = p.parse_args()
    AutoDrivePTQ(args).run()


if __name__ == "__main__":
    main()
