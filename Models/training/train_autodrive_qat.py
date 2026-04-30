"""
AutoDrive QAT — Quantization-Aware Training + INT8 ONNX export + benchmark.

One script, one run.

Usage
─────
  # Train only
  python Models/training/train_autodrive_qat.py \\
      --root ~/data/zod \\
      --checkpoint ~/data/zod/training/autodrive/run002/checkpoints/AutoDrive_best.pth

  # Train → convert → export ONNX → benchmark
  python Models/training/train_autodrive_qat.py \\
      --root ~/data/zod \\
      --checkpoint ~/data/zod/training/autodrive/run002/checkpoints/AutoDrive_best.pth \\
      --export-onnx --benchmark

Outputs  →  {root}/training/autodrive_qat/{run_name}/
  checkpoints/AutoDrive_qat_last.pth        QAT prepared  (latest, resumable)
  checkpoints/AutoDrive_qat_best.pth        QAT prepared  (best val loss)
  checkpoints/AutoDrive_qat_converted.pth   INT8 converted (from best)
  checkpoints/AutoDrive_qat_int8.onnx       INT8 ONNX      (with --export-onnx)
  tensorboard/

Notes
─────
  • XNNPACKQuantizer, per-tensor (is_per_channel=False).
    Per-tensor is the only config the ONNX exporter supports.
  • convert_pt2e is called ONCE at the end on the best checkpoint.
    Calling it every epoch gives bad results (observers still calibrating).
  • ONNX export: calibrates with val data, then dynamo-exports.
"""

import copy, math, sys
from argparse import ArgumentParser
from pathlib import Path

import torch, torch.nn as nn, tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)
import torchao.quantization.pt2e as pt2e_utils
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive, CURV_SCALE
from Models.training.auto_drive_trainer import AverageMeter


# ── constants (same as AutoDriveTrainer) ────────────────────────────────────
_POS_W   = torch.tensor([0.295 / 0.705])
_CW, _DW = 2.0, 2.0          # curvature / distance loss weights
_THR     = 0.65               # flag classification threshold
_WB      = 2.984              # wheelbase (m)
_SCR     = 16.8               # steering column ratio


def _collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


# ── loss + metrics ───────────────────────────────────────────────────────────

def _loss(out, d_gt, c_gt, f_gt, mask, l1, bce, dev):
    d, c, fl = out
    lc = l1(c, c_gt)
    ld = l1(d[mask.unsqueeze(1)], d_gt[mask.unsqueeze(1)]) if mask.any() else torch.tensor(0., device=dev)
    lf = bce(fl, f_gt)
    return _CW * lc + _DW * ld + lf, lc.item(), ld.item(), lf.item()


@torch.no_grad()
def _metrics(out, d_gt, c_gt, f_gt, mask, l1, bce, dev):
    d, c, fl = out
    lc = l1(c, c_gt)
    if mask.any():
        mi = mask.unsqueeze(1)
        ld = l1(d[mi], d_gt[mi]);  dm = (150.*(d[mi]-d_gt[mi]).abs()).mean().item()
    else:
        ld = torch.tensor(0., device=dev); dm = 0.
    lf    = bce(fl, f_gt)
    total = lc + ld + lf
    facc  = ((torch.sigmoid(fl) > _THR).float() == f_gt).float().mean().item() * 100.
    sc    = (c * CURV_SCALE);  sg = (c_gt * CURV_SCALE);  ss = _SCR * (180./math.pi)
    smae  = (torch.atan(sc*_WB)*ss - torch.atan(sg*_WB)*ss).abs().mean().item()
    return dict(total=total.item(), curv=lc.item(), dist=ld.item(), flag=lf.item(),
                flag_acc=facc, dist_mae_m=dm, steer_mae_deg=smae)


@torch.no_grad()
def _val(model, loader, l1, bce, dev, label=""):
    keys = ("total","curv","dist","flag","flag_acc","dist_mae_m","steer_mae_deg")
    S = {k: 0. for k in keys};  n = 0
    for b in tqdm.tqdm(loader, desc=f"  val [{label}]", leave=False):
        out = model(b["img_prev"].to(dev), b["img_curr"].to(dev))
        m   = _metrics(out, b["d_norm"].unsqueeze(1).to(dev),
                       b["curvature"].unsqueeze(1).to(dev),
                       b["flag"].unsqueeze(1).to(dev),
                       b["dist_mask"].to(dev), l1, bce, dev)
        for k in keys: S[k] += m[k]
        n += 1
    return {k: S[k]/max(n,1) for k in keys}


# ── export + prepare (CPU, then move to device) ──────────────────────────────

def _prepare(device):
    """Export float AutoDrive, insert XNNPACK fake-quant nodes, move to device."""
    m = AutoDrive().eval().cpu()
    ex = (torch.randn(2,3,512,1024), torch.randn(2,3,512,1024))
    bd = torch.export.Dim("batch", min=2, max=128)
    exported = torch.export.export(
        m, ex, dynamic_shapes=({0:bd},{0:bd}), strict=False
    ).module()
    q = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
    )
    prepared = prepare_qat_pt2e(exported, q)   # CPU
    prepared.to(device)
    return prepared


# ── ONNX export ───────────────────────────────────────────────────────────────

def _export_onnx(best_ckpt, val_loader, onnx_path, calib_batches=100):
    """
    Load best QAT checkpoint → calibrate on val data (CPU) → convert_pt2e → ONNX.
    Returns onnx_path if successful, None otherwise.
    """
    print("\n  Building INT8 model for ONNX export …")
    prepared = _prepare(torch.device("cpu"))
    sd = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    prepared.load_state_dict(sd, strict=False)   # strict=False: observer shapes may differ
    pt2e_utils.move_exported_model_to_eval(prepared)

    print(f"  Calibrating ({calib_batches} batches) …")
    n = 0
    with torch.no_grad():
        for b in tqdm.tqdm(val_loader, total=calib_batches, leave=False):
            prepared(b["img_prev"], b["img_curr"])
            n += 1
            if n >= calib_batches:
                break

    int8 = convert_pt2e(copy.deepcopy(prepared))
    pt2e_utils.move_exported_model_to_eval(int8)

    print(f"  Exporting ONNX → {onnx_path} …")
    img = torch.randn(1, 3, 512, 1024)
    torch.onnx.export(
        int8, (img, img), str(onnx_path),
        dynamo=True,
        input_names=["image_prev", "image_curr"],
        output_names=["distance", "curvature", "flag_logit"],
        dynamic_shapes={
            "image_prev": {0: torch.export.Dim("batch", min=1, max=64)},
            "image_curr": {0: torch.export.Dim("batch", min=1, max=64)},
        },
    )
    mb = onnx_path.stat().st_size / 1e6
    print(f"  Saved ONNX ({mb:.1f} MB)")
    return onnx_path, int8


# ── benchmark: FP32 vs INT8 ORT ───────────────────────────────────────────────

import time as _time

@torch.no_grad()
def _benchmark(fp32_ckpt, int8_model, onnx_path, val_loader, device):
    import numpy as np
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        has_ort = True
    except ImportError:
        print("  onnxruntime not installed — skipping ORT benchmark.")
        has_ort = False

    fp32 = AutoDrive()
    ck   = torch.load(fp32_ckpt, map_location="cpu", weights_only=False)
    fp32.load_state_dict(ck["model"] if "model" in ck else ck)
    fp32.eval().to(device)

    l1  = nn.L1Loss(); bce = nn.BCEWithLogitsLoss(pos_weight=_POS_W)
    keys = ("total","curv","dist","flag","flag_acc","dist_mae_m","steer_mae_deg")
    S32  = {k: 0. for k in keys};  t32  = 0.
    Sort = {k: 0. for k in keys};  tort = 0.
    n = 0

    for b in tqdm.tqdm(val_loader, desc="  benchmark", leave=True):
        ip = b["img_prev"]; ic = b["img_curr"]
        dg = b["d_norm"].unsqueeze(1); cg = b["curvature"].unsqueeze(1)
        fg = b["flag"].unsqueeze(1);   mk = b["dist_mask"]

        t0 = _time.perf_counter()
        out32 = fp32(ip.to(device), ic.to(device))
        t32  += (_time.perf_counter()-t0)*1000.

        m32 = _metrics(tuple(x.cpu() for x in out32), dg, cg, fg, mk, l1, bce, torch.device("cpu"))
        for k in keys: S32[k] += m32[k]

        if has_ort:
            t0 = _time.perf_counter()
            oo = sess.run(None, {"image_prev": ip.numpy(), "image_curr": ic.numpy()})
            tort += (_time.perf_counter()-t0)*1000.
            ort_out = tuple(torch.from_numpy(x) for x in oo)
            mo = _metrics(ort_out, dg, cg, fg, mk, l1, bce, torch.device("cpu"))
            for k in keys: Sort[k] += mo[k]

        n += 1

    for d in (S32, Sort):
        for k in keys: d[k] /= max(n,1)

    w = 18; sep = "─"*(30+w*(2 if has_ort else 1))
    print(f"\n  ┌{sep}┐")
    print(f"  │  {'FP32 vs INT8 ONNX benchmark':<{28+w*(2 if has_ort else 1)}}│")
    print(f"  ├{sep}┤")
    hdr = f"  │  {'Metric':<28}{'FP32 PyTorch':>{w}}"
    if has_ort: hdr += f"{'INT8 ORT':>{w}}"
    print(hdr+"  │")
    print(f"  ├{sep}┤")
    rows=[("Val loss","total",".4f"),("Curvature","curv",".4f"),("Distance","dist",".4f"),
          ("Flag loss","flag",".4f"),("Steer MAE (°)","steer_mae_deg",".2f"),
          ("Dist MAE (m)","dist_mae_m",".2f"),("Flag acc (%)","flag_acc",".2f")]
    for nm,k,f in rows:
        line=f"  │  {nm:<28}{S32[k]:>{w}{f}}"
        if has_ort: line+=f"{Sort[k]:>{w}{f}}"
        print(line+"  │")
    fps32 = t32/max(n,1)/b["img_prev"].size(0)
    fport = tort/max(n,1)/b["img_prev"].size(0) if has_ort else 0.
    print(f"  │  {'ms / frame':<28}{fps32:>{w}.2f}"+(f"{fport:>{w}.2f}" if has_ort else "")+"  │")
    print(f"  └{sep}┘\n")


# ── LR schedule ──────────────────────────────────────────────────────────────

def _lr(epoch): return 1e-5 if epoch < 5 else 5e-6


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = ArgumentParser()
    ap.add_argument("--root",        required=True)
    ap.add_argument("--checkpoint",  required=True, help="Float AutoDrive .pth to start from")
    ap.add_argument("--run-name",    default="")
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--batch-size",  type=int, default=8)
    ap.add_argument("--workers",     type=int, default=2)
    ap.add_argument("--lr",          type=float, default=0., help="Override LR (0=schedule)")
    ap.add_argument("--obs-freeze",  type=int, default=4,  help="Freeze observers after epoch N")
    ap.add_argument("--bn-freeze",   type=int, default=3,  help="Freeze BN stats after epoch N")
    ap.add_argument("--export-onnx", action="store_true",  help="Convert + export INT8 ONNX after training")
    ap.add_argument("--benchmark",   action="store_true",  help="Benchmark FP32 vs INT8 ORT after export")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  AutoDrive QAT — {dev}\n{'='*60}\n")

    # directories
    base = Path(args.root)/"training"/"autodrive_qat"
    if args.run_name:
        rname = args.run_name
    else:
        existing = sorted(base.glob("qat[0-9][0-9][0-9]"))
        rname = f"qat{len(existing)+1:03d}"
    run_dir  = base/rname
    ckpt_dir = run_dir/"checkpoints"
    tb_dir   = run_dir/"tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    ckpt_last  = ckpt_dir/"AutoDrive_qat_last.pth"
    ckpt_best  = ckpt_dir/"AutoDrive_qat_best.pth"
    ckpt_conv  = ckpt_dir/"AutoDrive_qat_converted.pth"
    onnx_path  = ckpt_dir/"AutoDrive_qat_int8.onnx"

    print(f"  Run  : {run_dir}")
    print(f"  TB   : tensorboard --logdir {tb_dir}\n")

    # data
    data = LoadDataAutoDrive(args.root)
    train_loader = DataLoader(data.train, args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=_collate, drop_last=True)
    val_loader   = DataLoader(data.val,   args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=_collate)
    print(f"  Train {len(data.train):,}  Val {len(data.val):,}\n")

    # float model
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    float_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    # export + prepare (CPU → move to dev)
    print("  Preparing QAT model …")
    model = _prepare(dev)
    # load float weights into prepared graph
    float_m = AutoDrive(); float_m.load_state_dict(float_sd)
    float_m.eval().to(dev)
    # copy float state dict into prepared model (strict=False is safe here)
    model.load_state_dict(float_sd, strict=False)
    print(f"  Model ready on {dev}\n")

    l1   = nn.L1Loss()
    bce  = nn.BCEWithLogitsLoss(pos_weight=_POS_W.to(dev))
    opt  = torch.optim.Adam(model.parameters(),
                             lr=args.lr if args.lr > 0 else _lr(0), weight_decay=1e-5)
    tb   = SummaryWriter(str(tb_dir))
    best = float("inf");  step = 0

    _BN = {torch.ops.aten._native_batch_norm_legit.default,
           torch.ops.aten.cudnn_batch_norm.default,
           torch.ops.aten._native_batch_norm_legit_no_training.default}

    for epoch in range(args.epochs):
        lr = args.lr if args.lr > 0 else _lr(epoch)
        for pg in opt.param_groups: pg["lr"] = lr

        if epoch >= args.obs_freeze:
            model.apply(pt2e_utils.disable_observer)
        if epoch >= args.bn_freeze:
            for n in model.graph.nodes:
                if n.target in _BN and len(n.args) > 5:
                    a = list(n.args); a[5] = False; n.args = tuple(a)
            model.recompile()

        pt2e_utils.move_exported_model_to_train(model)
        avg = AverageMeter(); ac = AverageMeter(); ad = AverageMeter(); af = AverageMeter()

        bar = tqdm.tqdm(train_loader, desc=f"  ep {epoch+1}/{args.epochs}")
        for b in bar:
            out = model(b["img_prev"].to(dev), b["img_curr"].to(dev))
            loss, lc, ld, lf = _loss(out,
                b["d_norm"].unsqueeze(1).to(dev), b["curvature"].unsqueeze(1).to(dev),
                b["flag"].unsqueeze(1).to(dev),   b["dist_mask"].to(dev), l1, bce, dev)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
            bs = b["img_prev"].size(0)
            avg.update(loss.item(),bs); ac.update(lc,bs); ad.update(ld,bs); af.update(lf,bs)
            step += 1
            tb.add_scalar("Loss/train_total",     loss.item(), step)
            tb.add_scalar("Loss/train_curvature",  lc, step)
            tb.add_scalar("Loss/train_distance",   ld, step)
            tb.add_scalar("Loss/train_flag",       lf, step)
            bar.set_postfix(loss=f"{avg.avg:.4f}", c=f"{ac.avg:.4f}",
                            d=f"{ad.avg:.4f}", f=f"{af.avg:.4f}", lr=f"{lr:.1e}")

        tb.add_scalar("Loss/train_avg", avg.avg, epoch+1)
        tb.add_scalar("Metrics/lr",     lr,      epoch+1)

        torch.save(model.state_dict(), ckpt_last)

        pt2e_utils.move_exported_model_to_eval(model)
        m = _val(model, val_loader, l1, bce, dev)

        tb.add_scalar("Loss/val_total",          m["total"],          epoch+1)
        tb.add_scalar("Loss/val_curvature",       m["curv"],           epoch+1)
        tb.add_scalar("Loss/val_distance",        m["dist"],           epoch+1)
        tb.add_scalar("Loss/val_flag",            m["flag"],           epoch+1)
        tb.add_scalar("Metrics/val_flag_acc",     m["flag_acc"],       epoch+1)
        tb.add_scalar("Metrics/val_dist_mae_m",   m["dist_mae_m"],     epoch+1)
        tb.add_scalar("Metrics/val_steer_mae_deg",m["steer_mae_deg"],  epoch+1)
        tb.flush()

        print(f"\n  ep {epoch+1}  loss {m['total']:.4f}  "
              f"steer {m['steer_mae_deg']:.2f}°  "
              f"dist {m['dist_mae_m']:.1f}m  "
              f"flag {m['flag_acc']:.1f}%")

        if m["total"] < best:
            best = m["total"]
            torch.save(model.state_dict(), ckpt_best)
            print(f"  ★ best → {ckpt_best.name}")

        pt2e_utils.move_exported_model_to_train(model)

    # ── final INT8 conversion ────────────────────────────────────────────
    print(f"\n{'='*60}\n  Converting best checkpoint to INT8 …")
    model.load_state_dict(torch.load(ckpt_best, weights_only=True))
    int8 = convert_pt2e(copy.deepcopy(model).cpu())
    pt2e_utils.move_exported_model_to_eval(int8)
    torch.save(int8.state_dict(), ckpt_conv)
    print(f"  Saved → {ckpt_conv.name}")

    m_int8 = _val(int8.to(dev), val_loader, l1, bce, dev, label="INT8")
    print(f"  [INT8 val]  loss {m_int8['total']:.4f}  "
          f"steer {m_int8['steer_mae_deg']:.2f}°  "
          f"dist {m_int8['dist_mae_m']:.1f}m  "
          f"flag {m_int8['flag_acc']:.1f}%")

    # ── ONNX export ──────────────────────────────────────────────────────
    if args.export_onnx:
        _, int8_onnx_model = _export_onnx(ckpt_best, val_loader, onnx_path)

        if args.benchmark:
            _benchmark(args.checkpoint, int8_onnx_model, onnx_path, val_loader, dev)

    tb.flush(); tb.close()
    print(f"\n  Done. Outputs: {run_dir}\n")


if __name__ == "__main__":
    main()
