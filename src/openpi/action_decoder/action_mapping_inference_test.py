#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval script (same data-loading as your trainer):
- Read (actions, actions_base) using make_local_action_dataloader()
- Load MLP from best_action_mapping.pth
- Compare pred(actions) vs actions_base
"""

import argparse
import inspect
import math
import pickle
from typing import Dict, Any, Tuple

import torch

from src.openpi.action_decoder.action_mapping_model import ActionMappingMLP
from src.openpi.action_decoder.local_parquet_action_loader import make_local_action_dataloader


def safe_torch_load(path: str, map_location="cpu") -> Dict[str, Any]:
    """
    PyTorch>=2.6 torch.load default weights_only=True.
    Your ckpt includes pickled objects (mapping_config), so we need weights_only=False.
    """
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    try:
        return torch.load(path, **kwargs)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"torch.load failed with UnpicklingError.\n"
            f"Checkpoint: {path}\n"
            f"Try running with PyTorch>=2.6 and ensure weights_only=False is used."
        ) from e


@torch.no_grad()
def evaluate(
    ckpt_path: str,
    parquet_root: str ,
    split: str,
    device: str,
    batch_size: int ,
    max_samples: int ,
    print_examples: int,
) -> None:
    # 1) load ckpt + model
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    if "mapping_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(f"Unexpected ckpt keys: {list(ckpt.keys())}")

    mapping_config = ckpt["mapping_config"]
    if parquet_root is None:
        parquet_root = getattr(mapping_config, "parquet_root", None)
    if not parquet_root:
        raise ValueError("parquet_root is None. Please pass --parquet_root explicitly.")

    if batch_size is None:
        batch_size = int(getattr(mapping_config, "batch_size", 32))
    parquet_root = '/home/lbycdy/.cache/huggingface/lerobot/lbycdy/libero_20251219actionV5'
    dev = torch.device(device)
    model = ActionMappingMLP(mapping_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    # 2) dataloader (exactly like trainer, but shuffle_buffer=0 for deterministic eval)
    loader = make_local_action_dataloader(
        parquet_root=parquet_root,
        split=split,                 # "train" or "val"
        batch_size=batch_size,
        max_samples=max_samples,
        shuffle_buffer=0,
        num_workers=0,
    )

    # 3) metrics accumulators
    n_samples = 0
    n_dims = None

    se_sum = 0.0          # sum squared error over all elements
    ae_sum = 0.0          # sum abs error over all elements
    se_dim = None         # per-dim sum squared error
    ae_dim = None         # per-dim sum abs error

    # baseline: identity (actions as prediction) vs actions_base
    se_sum_id = 0.0
    ae_sum_id = 0.0

    shown = 0

    for actions, actions_base in loader:
        actions = actions.to(dev)
        actions_base = actions_base.to(dev)

        if actions.ndim != 2 or actions_base.ndim != 2:
            raise RuntimeError(f"Expected 2D tensors, got {actions.shape} and {actions_base.shape}")

        if n_dims is None:
            n_dims = actions_base.shape[1]

        pred = model(actions)

        diff = pred - actions_base
        se = diff.pow(2)
        ae = diff.abs()

        se_sum += float(se.sum().item())
        ae_sum += float(ae.sum().item())

        se_dim_batch = se.sum(dim=0).detach().cpu()
        ae_dim_batch = ae.sum(dim=0).detach().cpu()
        if se_dim is None:
            se_dim = se_dim_batch.clone()
            ae_dim = ae_dim_batch.clone()
        else:
            se_dim += se_dim_batch
            ae_dim += ae_dim_batch

        # identity baseline
        diff_id = actions - actions_base
        se_sum_id += float(diff_id.pow(2).sum().item())
        ae_sum_id += float(diff_id.abs().sum().item())

        # print examples
        if print_examples > 0 and shown < print_examples:
            bsz = actions.shape[0]
            k = min(print_examples - shown, bsz)
            a_np = actions[:k].detach().cpu().numpy()
            y_np = actions_base[:k].detach().cpu().numpy()
            p_np = pred[:k].detach().cpu().numpy()
            for i in range(k):
                err = p_np[i] - y_np[i]
                print(f"\n[Example {shown+1}]")
                print(f"  actions (cam):     {a_np[i]}")
                print(f"  gt actions_base:   {y_np[i]}")
                print(f"  pred actions_base: {p_np[i]}")
                print(f"  err (pred-gt):     {err}")
                print(f"  err L2:            {float((err**2).sum() ** 0.5):.6f}")
                shown += 1

        n_samples += actions.shape[0]

    if n_samples == 0:
        print("No samples were evaluated. Check split/max_samples/parquet_root.")
        return

    denom = n_samples * n_dims
    mse = se_sum / denom
    rmse = math.sqrt(mse)
    mae = ae_sum / denom

    mse_id = se_sum_id / denom
    rmse_id = math.sqrt(mse_id)
    mae_id = ae_sum_id / denom

    se_dim_np = (se_dim.numpy() / n_samples)  # per-dim MSE *? actually mean of squared error per dim
    ae_dim_np = (ae_dim.numpy() / n_samples)
    rmse_dim_np = (se_dim_np ** 0.5)

    print("\n================= Action Mapping Eval =================")
    print(f"ckpt       : {ckpt_path}")
    print(f"parquet_root: {parquet_root}")
    print(f"split      : {split}")
    print(f"device     : {device}")
    print(f"samples    : {n_samples}")
    print("------------------------------------------------------")
    print(f"[MODEL]   MSE : {mse:.8f}  RMSE: {rmse:.8f}  MAE: {mae:.8f}")
    print(f"[IDENTITY]MSE : {mse_id:.8f}  RMSE: {rmse_id:.8f}  MAE: {mae_id:.8f}")
    print("------------------------------------------------------")
    print("Per-dim RMSE:", rmse_dim_np)
    print("Per-dim MAE :", ae_dim_np)
    print("======================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/lbycdy/work/camerapi/best_action_mapping.pth")
    parser.add_argument("--parquet_root", type=str, default=None,
                        help="If None, use ckpt.mapping_config.parquet_root")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="If None, use ckpt.mapping_config.batch_size")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--print_examples", type=int, default=5)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        args.device = "cpu"

    evaluate(
        ckpt_path=args.ckpt,
        parquet_root=args.parquet_root,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        print_examples=args.print_examples,
    )


if __name__ == "__main__":
    main()
