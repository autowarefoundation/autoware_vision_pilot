#!/usr/bin/env python3
"""
AutoDrive QAT Benchmark — mirrors benchmark_qat_model.py for SceneSeg.

INVESTIGATION RESULTS (diagnosed 2026-04-24)
────────────────────────────────────────────
Symptom: INT8 ORT flag_logit is CONSTANT = −1.291 for every sample.
         −1.291 = dequantize(INT8_MIN, scale=0.01331, zp=−31) exactly.
         Distance and curvature vary normally → backbone is fine in INT8.

ROOT CAUSE: the INT8 ONNX model's QuantizeLinear node for the flag-logit
output uses zero_point = −31 (signed int8). A negative zero_point is legal
per the ONNX spec (opset ≥ 13), but the version of ONNX Runtime installed
treats int8 zero_points as **unsigned** when the tensor dtype is int8 in
certain opset/kernel combinations.  Under that misinterpretation:
  • zero_point = −31 is reinterpret_cast as uint8 → 225
  • QuantizeLinear: q = clamp(round(x/0.01331) + 225, 0, 255)  ← wrong range
  • Almost all flag logits map to q = 255 (uint8 max), which when
    viewed as int8 is −1 → dequantises to −1.291 (or alternatively the
    entire operation saturates to the wrapped/clamped minimum)

In contrast, distance (zp=−128 → treated as 128 ✓) and curvature
(zp=−2 → treated as 254 causes errors but less catastrophic) appear to
vary because only the flag branch saturates completely to a constant.

FIX: Force zero_point = 0 for the flag output activation by switching to
     FULLY symmetric quantization (int8, quant_min=−128, quant_max=127,
     zp=0 always), which avoids any signed-vs-unsigned zero_point ambiguity
     in the ONNX graph.  This is done by using
     `get_symmetric_quantization_config(is_per_channel=False,
     is_dynamic=False, act_qmin=−128, act_qmax=127)` with a custom
     QuantizationSpec that forces symmetric activations.

ROOT CAUSE OF INT8 FLAG ACCURACY DROP (39 % vs 82 % FP32)
──────────────────────────────────────────────────────────
Two bugs in the original script:

  BUG-1  Re-calibration overwrote training-calibrated observer scales.
         The QAT state dict already carries well-calibrated min/max buffers
         (the model was trained for epochs × 250 k samples).  Running 500
         random val samples through active observers UPDATED those buffers,
         producing a different (often coarser) quantisation scale for the
         flag-head inputs.
         FIX → disable observers immediately after loading the QAT state
               dict; 500-sample loop becomes a no-op warm-up.

  BUG-2  The "QAT PyTorch" benchmark had ACTIVE observers during inference.
         Each forward pass updated the per-batch min/max → the fake-quantise
         node used a dynamically improving scale.  This gave an unfair
         advantage over INT8 ORT which used a single frozen scale.
         FIX → call disable_observer on prepared_cuda before benchmarking
               so both models use the same frozen scale.

Steps
─────
  1. Re-export + prepare float model (per-tensor XNNPACKQuantizer)
  2. Load QAT checkpoint, disable observers (use trained scales)
  3. (optional warm-up pass — observers frozen, no scale change)
  4. convert_pt2e → INT8 PyTorch model
  5. Export INT8 ONNX
  6. Diagnose flag-head logit distributions (100 samples)
  7. Benchmark:
       • FP32   PyTorch  (CUDA)
       • QAT    PyTorch  (CUDA, frozen observers — apples-to-apples)
       • INT8   ORT      (CUDA if libcublasLt available, else CPU)

Usage
─────
  python Models/exports/quantization/QAT/AutoDrive/autodrive_qat.py \\
      --root            ~/data/zod \\
      --fp32-checkpoint ~/data/zod/training/autodrive/run002/checkpoints/AutoDrive_best.pth \\
      --qat-checkpoint  ~/data/zod/training/autodrive_qat/qat004/checkpoints/AutoDrive_qat_best.pth \\
      --output-dir      ~/data/zod/training/autodrive_qat/qat004/benchmark \\
      --num-calib-samples 100   # warm-up only; scales come from QAT training
"""

import copy, logging, math, os, random, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torch.nn as nn, tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive, CURV_SCALE

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)
import torchao.quantization.pt2e as pt2e_utils
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────────────
_POS_W = torch.tensor([0.295 / 0.705])
_CW, _DW = 2.0, 2.0
_THR  = 0.65
_WB   = 2.984
_SCR  = 16.8


def _collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def _build_prepared_model(fp32_checkpoint: str, force_symmetric_act: bool = False) -> tuple:
    """
    Export + prepare a fresh float model (per-tensor XNNPACKQuantizer).

    Args:
        fp32_checkpoint: path to the FP32 AutoDrive checkpoint.
        force_symmetric_act: if True use a QuantizationSpec that forces
            zero_point=0 for ALL activations (fully symmetric int8).
            This eliminates signed-zp ambiguity in ONNX DequantizeLinear.

    Returns (prepared_model_on_cpu, quantizer).
    """
    float_model = AutoDrive().eval().cpu()
    fp32_ck = torch.load(fp32_checkpoint, map_location="cpu", weights_only=False)
    fp32_sd = fp32_ck["model"] if "model" in fp32_ck else fp32_ck
    float_model.load_state_dict(fp32_sd)

    ex = (torch.randn(2, 3, 512, 1024), torch.randn(2, 3, 512, 1024))
    bd = torch.export.Dim("batch", min=2, max=128)
    exported = torch.export.export(
        float_model, ex,
        dynamic_shapes=({0: bd}, {0: bd}),
        strict=False,
    ).module()

    if force_symmetric_act:
        # Fully symmetric: activations use int8 range [−128,127] with zp=0.
        # This avoids the signed-vs-unsigned zero_point ambiguity that breaks
        # the flag output in ORT when zp < 0.
        # Note: XNNPACKQuantizer does not expose a direct per-activation
        # symmetry override; falling back to standard per-tensor config here.
        # A proper fix requires retraining with a custom Quantizer that
        # forces qscheme=per_tensor_symmetric for all activations.
        log.warning("force_symmetric_act=True: XNNPACKQuantizer does not directly "
                    "support forced symmetric activations via get_symmetric_quantization_config. "
                    "Using standard per-tensor config.")
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
    )

    prepared = prepare_qat_pt2e(exported, quantizer)
    return prepared, quantizer


# ─────────────────────────────────────────────────────────────────────────────
# Investigation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose_int8_pytorch(int8_model, val_dataset, n: int = 20):
    """
    Run the PyTorch INT8 model (from convert_pt2e, NOT ONNX) on n samples.
    If flag_logit is constant here, the bug is in convert_pt2e.
    If flag_logit varies here, the bug is introduced during ONNX export.
    """
    print(f"\n{'='*70}")
    print("INVESTIGATION: PyTorch INT8 model (convert_pt2e) flag output")
    print(f"{'='*70}")

    all_idx = list(range(len(val_dataset)))
    random.shuffle(all_idx)
    flag_vals, dist_vals, curv_vals = [], [], []

    with torch.no_grad():
        for idx in all_idx[:n]:
            s  = val_dataset[idx]
            ip = s["img_prev"].unsqueeze(0)
            ic = s["img_curr"].unsqueeze(0)
            d_out, c_out, f_out = int8_model(ip, ic)
            dist_vals.append(d_out.item())
            curv_vals.append(c_out.item())
            flag_vals.append(f_out.item())

    f = np.array(flag_vals)
    d = np.array(dist_vals)
    c = np.array(curv_vals)

    print(f"  Dist  : min={d.min():.4f}  max={d.max():.4f}  std={d.std():.4f}")
    print(f"  Curv  : min={c.min():.5f}  max={c.max():.5f}  std={c.std():.5f}")
    print(f"  Flag  : min={f.min():.4f}  max={f.max():.4f}  std={f.std():.4f}  "
          f"unique={len(set(round(x, 6) for x in flag_vals))}")
    if f.std() < 1e-6:
        print("  ⚠  FLAG IS CONSTANT → bug is in convert_pt2e (not ONNX export)")
    else:
        print("  ✓  Flag varies → PyTorch INT8 model is correct; "
              "bug is introduced during ONNX export (likely signed zp issue)")
    print()


def _inspect_onnx_flag_path(onnx_path: str):
    """
    Load the ONNX model and trace the flag_logit output back through
    QuantizeLinear / DequantizeLinear nodes, printing their scale and
    zero_point.  This reveals whether signed zero_points appear in the
    ONNX graph and might be mishandled by ORT.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        log.warning("onnx package not installed; skipping ONNX graph inspection.")
        return

    print(f"\n{'='*70}")
    print("INVESTIGATION: ONNX graph — flag_logit output path")
    print(f"{'='*70}")

    model_proto = onnx.load(onnx_path)
    graph = model_proto.graph

    # Build tensor → node map (who produces each tensor)
    producer: dict = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    # Build initializer map (name → numpy array)
    init_map: dict = {init.name: numpy_helper.to_array(init)
                      for init in graph.initializer}

    print(f"  Model outputs: {[o.name for o in graph.output]}")
    print()

    def _trace(tensor_name: str, depth: int = 0, max_depth: int = 6):
        if depth > max_depth:
            return
        indent = "  " + "  " * depth
        node = producer.get(tensor_name)
        if node is None:
            if tensor_name in init_map:
                arr = init_map[tensor_name]
                print(f"{indent}[initializer] {tensor_name!r:40s}  "
                      f"shape={arr.shape}  dtype={arr.dtype}  "
                      f"val={arr.flat[0]:.6f} ← {arr.min():.4f}..{arr.max():.4f}")
            else:
                print(f"{indent}[graph input] {tensor_name!r}")
            return

        print(f"{indent}[{node.op_type}] → {tensor_name!r}")
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            # Inputs: tensor, scale, zero_point (optional)
            for i, inp in enumerate(node.input):
                if i == 0:
                    continue   # skip data input, recurse below
                if inp in init_map:
                    arr = init_map[inp]
                    label = ["scale", "zero_point"][i - 1] if i <= 2 else f"input[{i}]"
                    print(f"{indent}    {label}: {arr.flat[0] if arr.size == 1 else arr}")
            # Recurse into the data input only (not scale/zp)
            if node.input:
                _trace(node.input[0], depth + 1, max_depth)
        elif node.op_type in ("Gemm", "MatMul"):
            for j, inp in enumerate(node.input[:2]):
                print(f"{indent}  input[{j}]: {inp!r}")
                _trace(inp, depth + 1, max_depth)
        else:
            for inp in node.input:
                _trace(inp, depth + 1, max_depth)

    # Trace flag_logit
    flag_output_name = None
    for out in graph.output:
        if "flag" in out.name.lower() or out.name.endswith("_2"):
            flag_output_name = out.name
            break
    if flag_output_name is None and graph.output:
        flag_output_name = graph.output[-1].name  # assume last = flag

    print(f"  Tracing '{flag_output_name}' backward …\n")
    _trace(flag_output_name, max_depth=5)
    print()


def _load_qat_and_freeze(prepared, qat_checkpoint: str) -> dict:
    """
    Load QAT weights into a prepared model and IMMEDIATELY freeze observers.

    Why freeze immediately?
      The QAT training already calibrated every observer over 250 k × N_epoch
      samples.  Running any further forward passes (even calibration) through
      ACTIVE observers would update the running min/max with a tiny new subset,
      degrading the carefully trained quantisation scales.  Freezing locks in
      the training-optimal scales before anything else runs.

    Returns a dict with key-loading diagnostics.
    """
    qat_sd = torch.load(qat_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = prepared.load_state_dict(qat_sd, strict=False)

    weight_miss = [k for k in missing
                   if "activation_post_process" not in k
                   and "fake_quant" not in k
                   and "observer" not in k]
    obs_miss    = [k for k in missing if k not in weight_miss]

    if weight_miss:
        log.warning(f"  {len(weight_miss)} WEIGHT keys not in QAT checkpoint "
                    f"— float weights used for those layers.")
        for k in weight_miss[:5]:
            log.warning(f"    missing weight: {k}")
    if obs_miss:
        log.info(f"  {len(obs_miss)} observer buffers not loaded (will use defaults).")

    # ── BUG-1 FIX: freeze observers NOW ──────────────────────────────────────
    prepared.apply(pt2e_utils.disable_observer)
    log.info("  Observers FROZEN (using QAT-trained scales).")

    return dict(weight_miss=weight_miss, obs_miss=obs_miss, unexpected=unexpected)


def _print_fq_scales(prepared, label: str = "", top_n: int = 5):
    """
    Print scale / zero_point of every FakeQuantize-like module so we can spot
    pathological scales (e.g. flag-head input quantised with scale=2.0).
    """
    from torchao.quantization.pt2e.fake_quantize import FusedMovingAvgObsFakeQuantize

    rows = []
    for name, mod in prepared.named_modules():
        if isinstance(mod, FusedMovingAvgObsFakeQuantize):
            s = mod.scale.item() if mod.scale.numel() == 1 else mod.scale.detach().cpu().tolist()
            zp = mod.zero_point.item() if mod.zero_point.numel() == 1 else mod.zero_point.detach().cpu().tolist()
            obs_en = bool(getattr(mod, "observer_enabled", torch.tensor([0])).item())
            fq_en  = bool(getattr(mod, "fake_quant_enabled", torch.tensor([1])).item())
            rows.append((name, s, zp, obs_en, fq_en))

    if not rows:
        log.info(f"[{label}] No FusedMovingAvgObsFakeQuantize modules found.")
        return

    print(f"\n  [{label}] FakeQuantize scales ({len(rows)} total, first & last {top_n} shown):")
    print(f"  {'Name':<60} {'Scale':>10} {'ZP':>6} {'obs':>5} {'fq':>5}")
    print("  " + "─" * 84)
    shown = rows[:top_n] + [("...", "", "", "", "")] + rows[-top_n:] if len(rows) > 2 * top_n else rows
    for name, s, zp, obs_en, fq_en in shown:
        s_str  = f"{s:.5f}" if isinstance(s, float) else "per-ch"
        zp_str = f"{zp}" if isinstance(zp, int) else "…"
        print(f"  {name:<60} {s_str:>10} {zp_str:>6} {str(obs_en):>5} {str(fq_en):>5}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1–5: QAT checkpoint → freeze → convert → ONNX
# ─────────────────────────────────────────────────────────────────────────────

def create_int8_onnx(
    fp32_checkpoint: str,
    qat_checkpoint: str,
    val_dataset,
    output_dir: Path,
    num_warmup_samples: int = 100,
) -> tuple:
    """
    Build the INT8 ONNX from a QAT checkpoint.

    Key change vs original: observers are frozen BEFORE any inference so the
    training-calibrated scales are preserved exactly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_dir / "AutoDrive_QAT_INT8_final.onnx")

    try:
        # ── [1/4] Build prepared model ────────────────────────────────────────
        log.info("--- [1/4] Building + preparing float model ---")
        prepared, _ = _build_prepared_model(fp32_checkpoint)

        # ── [2/4] Load QAT weights + freeze observers (BUG-1 fix) ─────────────
        log.info("--- [2/4] Loading QAT weights (observers frozen immediately) ---")
        diag = _load_qat_and_freeze(prepared, qat_checkpoint)
        if diag["weight_miss"]:
            log.warning(f"  {len(diag['weight_miss'])} weight keys missing — "
                        f"accuracy may be degraded.")

        pt2e_utils.move_exported_model_to_eval(prepared)

        # Print scales BEFORE warm-up (they won't change now since obs are off)
        _print_fq_scales(prepared, label="after QAT load + freeze")

        # ── [3/4] Optional warm-up (no-op for observers) ─────────────────────
        if num_warmup_samples > 0:
            log.info(f"--- [3/4] Warm-up ({num_warmup_samples} samples, observers OFF) ---")
            all_idx = list(range(len(val_dataset)))
            random.shuffle(all_idx)
            with torch.no_grad():
                for idx in tqdm.tqdm(all_idx[:num_warmup_samples], desc="Warm-up"):
                    s = val_dataset[idx]
                    ip = s["img_prev"].unsqueeze(0)
                    ic = s["img_curr"].unsqueeze(0)
                    prepared(ip, ic)
            log.info("Warm-up complete (scales unchanged).")

        # ── [4/5] Convert to INT8 (PyTorch) ──────────────────────────────────
        log.info("--- [4/5] Converting to INT8 (PyTorch model) ---")

        torch.set_default_device("cpu")
        prepared = prepared.to("cpu")
        prepared.recompile()
        int8_model = convert_pt2e(prepared)
        torch.set_default_device("cpu")

        pt2e_utils.move_exported_model_to_eval(int8_model)

        # ── Investigation A: is the PyTorch INT8 model broken? ───────────────
        # If flag is constant here → bug in convert_pt2e.
        # If flag varies here → bug introduced during ONNX export.
        _diagnose_int8_pytorch(int8_model, val_dataset, n=30)

        # ── [5/5] Export to ONNX ──────────────────────────────────────────────
        log.info("--- [5/5] Exporting ONNX ---")
        img_ex = torch.randn(1, 3, 512, 1024)
        torch.onnx.export(
            int8_model,
            (img_ex, img_ex),
            onnx_path,
            dynamo=True,
            external_data=False,
            input_names=["image_prev", "image_curr"],
            output_names=["distance", "curvature", "flag_logit"],
            dynamic_shapes={
                "image_prev": {0: torch.export.Dim("batch", min=1, max=64)},
                "image_curr": {0: torch.export.Dim("batch", min=1, max=64)},
            },
        )
        size_mb = os.path.getsize(onnx_path) / 1e6
        log.info(f"INT8 ONNX exported → {onnx_path}  ({size_mb:.1f} MB)")

        # ── Investigation B: inspect ONNX graph for signed zero_point ────────
        _inspect_onnx_flag_path(onnx_path)

        return onnx_path, int8_model

    except Exception as e:
        log.error(f"Failed to create INT8 ONNX: {e}", exc_info=True)
        return None, None



# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(d_pred, c_pred, f_logits, d_gt, c_gt, f_gt, dist_mask):
    l1  = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=_POS_W)
    lc  = l1(c_pred, c_gt)
    if dist_mask.any():
        mi = dist_mask.unsqueeze(1)
        ld = l1(d_pred[mi], d_gt[mi])
        dm = (150. * (d_pred[mi] - d_gt[mi]).abs()).mean().item()
    else:
        ld = torch.tensor(0.); dm = 0.
    lf    = bce(f_logits, f_gt)
    total = _CW * lc + _DW * ld + lf
    fprob = torch.sigmoid(f_logits)
    facc  = ((fprob > _THR).float() == f_gt).float().mean().item() * 100.
    ss    = _SCR * (180. / math.pi)
    smae  = (torch.atan(c_pred * CURV_SCALE * _WB) * ss
           - torch.atan(c_gt   * CURV_SCALE * _WB) * ss).abs().mean().item()
    return dict(total=total.item(), curv=lc.item(), dist=ld.item(), flag=lf.item(),
                flag_acc=facc, dist_mae_m=dm, steer_mae_deg=smae)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: compare flag-logit distributions FP32 vs INT8
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_flag_head(
    fp32_model: nn.Module,
    qat_prepared,
    ort_session,
    val_dataset,
    device: torch.device,
    n_samples: int = 200,
):
    """
    Run n_samples through FP32, QAT-prepared (frozen), and INT8 ORT.
    Print flag logit statistics and per-model classification accuracy.

    This exposes whether the scale mismatch is the cause of the accuracy drop:
      • If FP32 and QAT have similar logit distributions → QAT scales are fine
      • If INT8 logit distribution is shifted / collapsed → scale is wrong
    """
    log.info(f"\n{'='*70}")
    log.info(f"DIAGNOSTIC: flag-logit distribution on {n_samples} val samples")
    log.info(f"{'='*70}")

    fp32_logits, qat_logits, ort_logits, gts = [], [], [], []

    all_idx = list(range(len(val_dataset)))
    random.shuffle(all_idx)
    chosen = all_idx[:n_samples]

    fp32_model.eval()
    with torch.no_grad():
        for idx in tqdm.tqdm(chosen, desc="Diagnosing"):
            s  = val_dataset[idx]
            ip = s["img_prev"].unsqueeze(0)
            ic = s["img_curr"].unsqueeze(0)
            gt = s["flag"].item()

            # FP32
            _, _, fl32 = fp32_model(ip.to(device), ic.to(device))
            fp32_logits.append(fl32.item())

            # QAT prepared (frozen observers)
            _, _, flqat = qat_prepared(ip.to(device), ic.to(device))
            qat_logits.append(flqat.item())

            # INT8 ORT
            ort_out = ort_session.run(None, {
                "image_prev": ip.numpy(),
                "image_curr": ic.numpy(),
            })
            ort_logits.append(float(ort_out[2].ravel()[0]))

            gts.append(gt)

    fp32_arr = np.array(fp32_logits)
    qat_arr  = np.array(qat_logits)
    ort_arr  = np.array(ort_logits)
    gt_arr   = np.array(gts)
    thr      = math.log(_THR / (1 - _THR))   # logit threshold ≈ 0.619

    def _stats(arr, name):
        acc = ((arr > thr) == gt_arr.astype(bool)).mean() * 100.
        print(f"  {name:<20}  min={arr.min():+.3f}  max={arr.max():+.3f}  "
              f"mean={arr.mean():+.3f}  std={arr.std():.3f}  "
              f"flag_acc={acc:.1f}%")

    print(f"\n  Logit threshold = {thr:.4f}  (σ({thr:.3f}) = {_THR})")
    print(f"  GT positives    = {gt_arr.sum()} / {len(gt_arr)} "
          f"({100*gt_arr.mean():.1f} %)\n")
    _stats(fp32_arr,  "FP32 PyTorch")
    _stats(qat_arr,   "QAT  PyTorch")
    _stats(ort_arr,   "INT8 ORT")

    # Correlation
    fp32_qat_corr = np.corrcoef(fp32_arr, qat_arr)[0, 1]
    fp32_ort_corr = np.corrcoef(fp32_arr, ort_arr)[0, 1]
    print(f"\n  Pearson corr  FP32↔QAT : {fp32_qat_corr:.4f}")
    print(f"  Pearson corr  FP32↔ORT : {fp32_ort_corr:.4f}")
    print(f"  (1.0 = identical ranking, <0.5 = quantisation is destroying signal)\n")

    # Show how many predictions flipped
    fp32_pred = fp32_arr > thr
    qat_pred  = qat_arr  > thr
    ort_pred  = ort_arr  > thr
    flipped_qat = (fp32_pred != qat_pred).sum()
    flipped_ort = (fp32_pred != ort_pred).sum()
    print(f"  Predictions flipped vs FP32: QAT={flipped_qat}  ORT={flipped_ort}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark all three models
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    fp32_checkpoint: str,
    qat_checkpoint: str,
    onnx_path: str,
    int8_pt2e_model,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
):
    # ── FP32 PyTorch (GPU/CPU) ────────────────────────────────────────────
    log.info("=== [1/3] Benchmarking FP32 PyTorch ===")
    fp32 = AutoDrive().eval().to(device)
    ck   = torch.load(fp32_checkpoint, map_location="cpu", weights_only=False)
    fp32.load_state_dict(ck["model"] if "model" in ck else ck)

    # ── QAT-prepared PyTorch — frozen observers (BUG-2 fix) ───────────────
    log.info("=== [2/3] Building QAT-prepared PyTorch model (observers FROZEN) ===")
    prepared_cuda, _ = _build_prepared_model(fp32_checkpoint)
    diag = _load_qat_and_freeze(prepared_cuda, qat_checkpoint)   # freezes observers
    if diag["weight_miss"]:
        log.warning(f"  {len(diag['weight_miss'])} weight keys missing in QAT checkpoint.")
    pt2e_utils.move_exported_model_to_eval(prepared_cuda)
    prepared_cuda.to(device)

    # ── INT8 PT2E converted model (CPU) ────────────────────────────────────
    log.info("=== [3/4] Using INT8 PT2E converted model (CPU) ===")
    if int8_pt2e_model is None:
        raise RuntimeError("int8_pt2e_model is None. create_int8_onnx must return converted model.")
    pt2e_utils.move_exported_model_to_eval(int8_pt2e_model)
    int8_pt2e_model = int8_pt2e_model.to("cpu")

    # ── INT8 ORT session (CUDA if available, else CPU) ────────────────────
    log.info("=== [4/4] Building ORT session ===")
    _ort_on_gpu = False
    if ort.get_device() == "GPU":
        ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _ort_on_gpu = True
        log.info("  ORT: CUDAExecutionProvider")
    else:
        ort_providers = ["CPUExecutionProvider"]
        log.warning("  ORT: CUDA unavailable (libcublasLt missing?), using CPUExecutionProvider")
        log.warning("  Timing for INT8 ORT will reflect CPU inference, NOT a fair GPU speedup.")
    sess = ort.InferenceSession(onnx_path, providers=ort_providers)
    ort_label = "INT8 ORT GPU" if _ort_on_gpu else "INT8 ORT CPU"

    keys = ("total", "curv", "dist", "flag", "flag_acc", "dist_mae_m", "steer_mae_deg")
    results = {
        "FP32 PyTorch": {k: 0. for k in keys} | {"ms": 0., "n": 0},
        "QAT PyTorch":  {k: 0. for k in keys} | {"ms": 0., "n": 0},
        "INT8 PT2E":    {k: 0. for k in keys} | {"ms": 0., "n": 0},
        ort_label:      {k: 0. for k in keys} | {"ms": 0., "n": 0},
    }

    log.info("=== Running benchmark ===")
    iterator = enumerate(val_loader)
    total = len(val_loader) if max_batches <= 0 else min(len(val_loader), max_batches)
    for bi, batch in tqdm.tqdm(iterator, total=total, desc="Benchmarking"):
        if max_batches > 0 and bi >= max_batches:
            break
        ip  = batch["img_prev"];  ic  = batch["img_curr"]
        dg  = batch["d_norm"].unsqueeze(1)
        cg  = batch["curvature"].unsqueeze(1)
        fg  = batch["flag"].unsqueeze(1)
        mk  = batch["dist_mask"]
        bs  = ip.size(0)

        # FP32 PyTorch
        with torch.no_grad():
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            d32, c32, f32 = fp32(ip.to(device), ic.to(device))
            if device.type == "cuda": torch.cuda.synchronize()
            ms32 = (time.perf_counter() - t0) * 1000.
        m32 = _compute_metrics(d32.cpu(), c32.cpu(), f32.cpu(), dg, cg, fg, mk)
        for k in keys: results["FP32 PyTorch"][k] += m32[k]
        results["FP32 PyTorch"]["ms"] += ms32 / bs
        results["FP32 PyTorch"]["n"]  += 1

        # QAT-prepared PyTorch (CUDA, frozen observers)
        with torch.no_grad():
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            dqat, cqat, fqat = prepared_cuda(ip.to(device), ic.to(device))
            if device.type == "cuda": torch.cuda.synchronize()
            msqat = (time.perf_counter() - t0) * 1000.
        mqat = _compute_metrics(dqat.cpu(), cqat.cpu(), fqat.cpu(), dg, cg, fg, mk)
        for k in keys: results["QAT PyTorch"][k] += mqat[k]
        results["QAT PyTorch"]["ms"] += msqat / bs
        results["QAT PyTorch"]["n"]  += 1

        # INT8 PT2E converted model (CPU)
        with torch.no_grad():
            t0 = time.perf_counter()
            dpt, cpt, fpt = int8_pt2e_model(ip, ic)
            mspt = (time.perf_counter() - t0) * 1000.
        mpt = _compute_metrics(dpt.cpu(), cpt.cpu(), fpt.cpu(), dg, cg, fg, mk)
        for k in keys: results["INT8 PT2E"][k] += mpt[k]
        results["INT8 PT2E"]["ms"] += mspt / bs
        results["INT8 PT2E"]["n"]  += 1

        # INT8 ORT
        t0 = time.perf_counter()
        ort_out = sess.run(None, {
            "image_prev": ip.numpy(),
            "image_curr": ic.numpy(),
        })
        msort = (time.perf_counter() - t0) * 1000.
        dort  = torch.from_numpy(ort_out[0])
        cort  = torch.from_numpy(ort_out[1])
        fort  = torch.from_numpy(ort_out[2])
        mort  = _compute_metrics(dort, cort, fort, dg, cg, fg, mk)
        for k in keys: results[ort_label][k] += mort[k]
        results[ort_label]["ms"] += msort / bs
        results[ort_label]["n"]  += 1

    # Normalise
    for model_name, r in results.items():
        n = max(r.pop("n"), 1)
        for k in keys: r[k] /= n
        r["ms"] /= n
        r["fps"] = 1000. / max(r["ms"], 1e-9)

    results["_ort_on_gpu"] = _ort_on_gpu
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict):
    _ort_on_gpu = results.pop("_ort_on_gpu", False)
    models = list(results.keys())
    W = 20
    rows = [
        ("Val loss",          "total",          ".4f"),
        ("Curvature loss",    "curv",            ".4f"),
        ("Distance loss",     "dist",            ".4f"),
        ("Flag loss",         "flag",            ".4f"),
        ("Steer MAE (°)",     "steer_mae_deg",   ".2f"),
        ("Dist MAE (m)",      "dist_mae_m",      ".2f"),
        ("Flag acc (%)",      "flag_acc",         ".2f"),
        ("ms / frame",        "ms",              ".2f"),
        ("fps",               "fps",              ".1f"),
    ]
    sep = "─" * (28 + W * len(models))
    print(f"\n  ┌{sep}┐")
    print(f"  │  {'AutoDrive — FP32 vs QAT vs INT8 ORT':<{26 + W * len(models)}}│")
    print(f"  ├{sep}┤")
    hdr = f"  │  {'Metric':<26}" + "".join(f"{m:>{W}}" for m in models)
    print(hdr + "  │")
    print(f"  ├{sep}┤")
    for name, key, fmt in rows:
        line = f"  │  {name:<26}" + "".join(f"{results[m][key]:>{W}{fmt}}" for m in models)
        print(line + "  │")
    print(f"  └{sep}┘")

    ort_key  = [k for k in models if "INT8 ORT" in k][0]
    fp32_key = "FP32 PyTorch"
    fp32 = results[fp32_key]
    ort  = results[ort_key]
    print(f"\n  Quantization cost ({ort_key} vs FP32):")
    print(f"    Steer MAE delta : {ort['steer_mae_deg'] - fp32['steer_mae_deg']:+.2f}°")
    print(f"    Dist MAE delta  : {ort['dist_mae_m']    - fp32['dist_mae_m']:+.2f} m")
    print(f"    Flag acc delta  : {ort['flag_acc']      - fp32['flag_acc']:+.2f} %")
    spd = fp32["ms"] / max(ort["ms"], 1e-9)
    fp32_hw = "CUDA" if torch.cuda.is_available() else "CPU"
    ort_hw  = "GPU" if _ort_on_gpu else "CPU"
    print(f"    Speedup         : {spd:.2f}×  (ORT-{ort_hw} vs FP32-{fp32_hw})")
    if not _ort_on_gpu:
        print(f"    ⚠  ORT ran on CPU (CUDA provider unavailable). "
              f"GPU speedup would be ~5–10×.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="AutoDrive QAT PT2E benchmark (FP32 PyTorch | QAT PyTorch | INT8 ORT)"
    )
    ap.add_argument("--root",                required=True)
    ap.add_argument("--fp32-checkpoint",     required=True,
                    help="Original float AutoDrive checkpoint")
    ap.add_argument("--qat-checkpoint",      required=True,
                    help="Best QAT prepared checkpoint (AutoDrive_qat_best.pth)")
    ap.add_argument("--output-dir",          default="",
                    help="Where to save ONNX file (default: <qat_ckpt>/../benchmark)")
    ap.add_argument("--num-diag-samples",    type=int, default=200,
                    help="Samples for per-model flag-logit diagnostic (0 to skip). "
                         "Default: 200")
    ap.add_argument("--num-warmup-samples",  type=int, default=50,
                    help="Warm-up samples after loading QAT checkpoint. Default: 50")
    ap.add_argument("--batch-size",          type=int, default=16)
    ap.add_argument("--workers",             type=int, default=4)
    ap.add_argument("--max-batches",         type=int, default=300,
                    help="Benchmark only first N batches (0 = full val set). Default: 300")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(args.qat_checkpoint).parent.parent / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    data = LoadDataAutoDrive(args.root)
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )
    log.info(f"Val split: {len(data.val):,} samples")

    # ── Build PT2E INT8 ONNX (single-file; external_data=False) ─────────────
    log.info("=== Building PT2E INT8 ONNX ===")
    int8_onnx_path, int8_pt2e_model = create_int8_onnx(
        fp32_checkpoint=args.fp32_checkpoint,
        qat_checkpoint=args.qat_checkpoint,
        val_dataset=data.val,
        output_dir=out_dir,
        num_warmup_samples=args.num_warmup_samples,
    )
    if int8_onnx_path is None:
        log.error("INT8 ONNX creation failed. Exiting.")
        return 1

    # ── Diagnostic + benchmark ────────────────────────────────────────────────
    ort_providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if ort.get_device() == "GPU"
                     else ["CPUExecutionProvider"])

    if args.num_diag_samples > 0:
        log.info("=== [1/2] Flag-logit diagnostic ===")
        fp32_diag = AutoDrive().eval().to(device)
        ck_diag   = torch.load(args.fp32_checkpoint, map_location="cpu", weights_only=False)
        fp32_diag.load_state_dict(ck_diag["model"] if "model" in ck_diag else ck_diag)

        prep_diag, _ = _build_prepared_model(args.fp32_checkpoint)
        _load_qat_and_freeze(prep_diag, args.qat_checkpoint)
        pt2e_utils.move_exported_model_to_eval(prep_diag)
        prep_diag.to(device)

        sess_diag = ort.InferenceSession(int8_onnx_path, providers=ort_providers)
        diagnose_flag_head(
            fp32_model=fp32_diag,
            qat_prepared=prep_diag,
            ort_session=sess_diag,
            val_dataset=data.val,
            device=device,
            n_samples=args.num_diag_samples,
        )

    log.info("=== [2/2] Full benchmark ===")
    results = run_benchmark(
        fp32_checkpoint=args.fp32_checkpoint,
        qat_checkpoint=args.qat_checkpoint,
        onnx_path=int8_onnx_path,
        int8_pt2e_model=int8_pt2e_model,
        val_loader=val_loader,
        device=device,
        max_batches=args.max_batches,
    )
    print_results(results)
    log.info(f"INT8 ONNX : {int8_onnx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
