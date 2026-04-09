import os
import re
import copy
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
import yaml
import numpy as np
from torch.utils import data

import auto_steer_util as util
from Models.data_utils.auto_steer.load_data_auto_steer import LoadDataAutoSteer
from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork

from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

loss_xp_weight = 10


def train(args, params, run_dir, log_writer):
    # Model
    model = AutoSteerNetwork().build_model(version=args.version)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    current_dir = Path(args.dataset + "/images/train/")
    filenames = [f.as_posix() for f in current_dir.rglob("*") if f.is_file()]

    sampler = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LoadDataAutoSteer(filenames)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                             num_workers=8, pin_memory=True)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    # criterion = util.ComputeLoss(model, params)
    l1 = torch.nn.L1Loss()
    bcel = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        if args.distributed:
            sampler.set_epoch(epoch)

        p_bar = enumerate(loader)

        if args.local_rank == 0:
            print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'xp', 'h_vector'))
            p_bar = tqdm.tqdm(p_bar, total=num_steps)

        optimizer.zero_grad()
        avg_loss_xp = util.AverageMeter()
        avg_loss_h_vector = util.AverageMeter()
        for i, (samples, targets_xp, targets_h_vector) in p_bar:
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)

            samples = samples.cuda().float() / 255
            targets_xp = targets_xp.cuda().float()
            targets_h_vector = targets_h_vector.cuda().float()

            # Forward
            with torch.amp.autocast('cuda'):
                output_xp, output_h_vector = model(samples)  # forward
                output_xp = output_xp * targets_h_vector
                loss_xp = loss_xp_weight * l1(output_xp, targets_xp)
                loss_h_vector = bcel(output_h_vector, targets_h_vector)

            avg_loss_xp.update(loss_xp.item(), samples.size(0))
            avg_loss_h_vector.update(loss_h_vector.item(), samples.size(0))

            loss_xp *= args.batch_size  # loss scaled by batch_size
            loss_h_vector *= args.batch_size  # loss scaled by batch_size

            # Backward
            amp_scale.scale(loss_xp + loss_h_vector).backward()

            # Optimize
            if step % accumulate == 0:
                # amp_scale.unscale_(optimizer)  # unscale gradients
                # util.clip_gradients(model)  # clip gradients
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            torch.cuda.synchronize()

            # Log
            if args.local_rank == 0:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 2) % (f'{epoch + 1}/{args.epochs}', memory, avg_loss_xp.avg,
                                                   avg_loss_h_vector.avg)
                p_bar.set_description(s)

            if i % 100 == 0:
                fig = util.visualize(samples, output_xp, output_h_vector, targets_xp, targets_h_vector)
                log_writer.add_figure('Train/Image', fig, global_step=(i))

        if args.local_rank == 0:
            # mAP
            last = val(args, params, run_dir, ema.ema)

            log_writer.add_scalar("Loss/xp", avg_loss_xp.avg, epoch + 1)
            log_writer.add_scalar("Loss/h_vector", avg_loss_h_vector.avg, epoch + 1)

            log_writer.add_scalar("Metrics/mAP", last[0], epoch + 1)
            log_writer.add_scalar("Metrics/mAP@50", last[1], epoch + 1)
            log_writer.add_scalar("Metrics/Recall", last[2], epoch + 1)
            log_writer.add_scalar("Metrics/Precision", last[3], epoch + 1)

            # Update best mAP
            if last[0] > best:
                best = last[0]

            # Save model
            save = {'epoch': epoch + 1,
                    'model': copy.deepcopy(ema.ema)}

            # Save last, best and delete
            torch.save(save, f=f'{run_dir}/weights/last.pt')
            if best == last[0]:
                torch.save(save, f=f'{run_dir}/weights/best.pt')
            del save

    if args.local_rank == 0:
        util.strip_optimizer(f'{run_dir}/weights/best.pt')  # strip optimizers
        util.strip_optimizer(f'{run_dir}/weights/last.pt')  # strip optimizers


@torch.no_grad()
def val(args, params, run_dir, model=None):
    current_dir = Path(args.dataset + "/images/val/")
    filenames = [f.as_posix() for f in current_dir.rglob("*") if f.is_file()]

    dataset = LoadDataAutoSteer(filenames)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    plot = False
    if not model:
        plot = True
        model = torch.load(f=f'{run_dir}/weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model = model.cuda().half()
    model.eval()

    # Configure multiple normalized thresholds
    thresholds = [0.02, 0.05, 0.10]  # example thresholds
    all_tp = {t: [] for t in thresholds}
    all_fp = {t: [] for t in thresholds}
    all_conf, all_output, all_target = [], [], []

    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP', 'mAP@50'))

    with torch.no_grad():
        for batch_idx, (samples, targets_xp, targets_h_vector) in enumerate(p_bar):
            samples = samples.cuda().half() / 255.0

            # Forward pass
            output_xp, output_h_vector = model(samples)

            batch_size = samples.shape[0]
            for i in range(batch_size):
                xp = output_xp[i]  # (2,64,1)
                h_vector = output_h_vector[i]  # (2,64,1)

                target_xp = targets_xp[i].cuda().float().view_as(xp)
                target_h_vector = targets_h_vector[i].cuda().float().view_as(h_vector)

                num_lines = xp.shape[0]
                for idx in range(num_lines):
                    # Apply mask (only valid lane points)
                    h_vector_mask = (h_vector[idx] > 0.5).float()
                    target_h_vector_mask = (target_h_vector[idx] > 0.5).float()

                    line = (xp[idx] * h_vector_mask).view(-1)
                    target_line = (target_xp[idx] * target_h_vector_mask).view(-1)

                    # Distance in normalized space
                    dist = torch.abs(line - target_line).mean().item()
                    conf = h_vector[idx].mean().item()

                    # Evaluate TP/FP for each threshold
                    for t in thresholds:
                        if dist < t:
                            all_tp[t].append(1)
                            all_fp[t].append(0)
                        else:
                            all_tp[t].append(0)
                            all_fp[t].append(1)

                    all_conf.append(conf)
                    all_output.append(line.cpu().numpy())
                    all_target.append(target_line.cpu().numpy())

                # Visualization
                global_step = batch_idx * batch_size + i
                if global_step % 100 == 0:
                    fig = util.visualize(samples, output_xp, output_h_vector, targets_xp, targets_h_vector)
                    log_writer.add_figure('Val/Image', fig, global_step=global_step)

    # Convert lists to arrays
    all_conf = np.array(all_conf)
    all_output = np.array(all_output)
    all_target = np.array(all_target)

    # Metrics
    if len(all_conf) > 0:
        m_pre, m_rec, mean_ap, map50 = util.compute_vector_ap(all_tp, all_fp, all_conf, all_target)
    else:
        print("No valid predictions found!")
        m_pre = m_rec = mean_ap = map50 = 0

    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, mean_ap, map50))
    model.float()
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 3, args.input_height, args.input_width)
    model = AutoSteerNetwork().build_model(version=args.version)
    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def get_next_run(path="."):
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Extract numbers from names like run1, run2, ...
    run_ids = []
    for d in subdirs:
        match = re.match(r"run(\d+)$", d)
        if match:
            run_ids.append(int(match.group(1)))

    if not run_ids:
        return 1  # If no runs exist yet, start with run1

    return max(run_ids) + 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', help="dataset directory path")
    parser.add_argument('--input-width', default=1024, type=int)
    parser.add_argument('--input-height', default=512, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=1, type=int)
    parser.add_argument('--version', default='n', type=str)
    # parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--runs_dir', default="runs/autosteer", type=str)
    parser.add_argument('--epochs', default=30, type=int)

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    # Prepare training directory
    if not os.path.exists(args.runs_dir):
        os.makedirs(args.runs_dir)
    next_run = get_next_run(args.runs_dir)
    run_dir = f"{args.runs_dir}/run{next_run}"
    weights_dir = f"{run_dir}/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    log_writer = SummaryWriter(log_dir=run_dir)

    torch.cuda.set_device(device=0)
    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('../config/auto_speed.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    train(args, params, run_dir, log_writer)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
