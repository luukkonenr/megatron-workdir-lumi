#!/usr/bin/env python
import argparse
import copy
import os
from collections import OrderedDict
import sys
sys.path.append("submodules/Megatron-LM")

import torch


###############################################################################
# Utilities
###############################################################################

def load_model_state(model_path):
    """Return ordered {name: tensor} for a TP rank, dropping extra_state."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    raw = ckpt["model"]  # your exploration indicates this is a dict
    # preserve order, drop extra_state
    state = OrderedDict((k, v) for k, v in raw.items() if "extra_state" not in k)
    return ckpt, state


def load_flat_optim(optim_path, group_idx=0,
                    dtype_pair=(torch.bfloat16, torch.float32)):
    """Return (opt_obj, group_dict, flat_param, flat_exp_avg, flat_exp_avg_sq)."""
    opt = torch.load(optim_path, map_location="cpu", weights_only=False)
    group = opt[group_idx][dtype_pair]
    flat_param = group["param"]
    flat_exp_avg = group["exp_avg"]
    flat_exp_avg_sq = group["exp_avg_sq"]
    return opt, group, flat_param, flat_exp_avg, flat_exp_avg_sq


def compute_slices(model_state):
    """Given ordered {name: tensor}, compute per-parameter numel and flat offsets."""
    names = list(model_state.keys())
    shapes = [t.shape for t in model_state.values()]
    numels = [t.numel() for t in model_state.values()]
    offsets = [0]
    for n in numels[:-1]:
        offsets.append(offsets[-1] + n)
    return names, shapes, numels, offsets
def get_shard_dim(name, tensor):
    # 1D params (LayerNorm, some biases, final_layernorm) are replicated
    if tensor.ndim == 1:
        return None

    # Input embeddings: vocab-parallel
    if "embedding.word_embeddings.weight" in name:
        return 0

    # Self-attention QKV: column-parallel
    if ".self_attention.linear_qkv.weight" in name:
        return 0

    # Self-attention output projection: row-parallel
    if ".self_attention.linear_proj.weight" in name:
        return 1

    # MLP up-projection: column-parallel
    if ".mlp.linear_fc1.weight" in name:
        return 0

    # MLP down-projection: row-parallel
    if ".mlp.linear_fc2.weight" in name:
        return 1

    # If you have a separate LM head later, you can add:
    # if "output_layer.weight" in name or "lm_head.weight" in name:
    #     return 0

    raise ValueError(f"Don't know shard dim for parameter {name} with shape {tuple(tensor.shape)}")

# def get_shard_dim(name, tensor):
#     """
#     Decide how to merge TP shards for this parameter.

#     Return:
#       - dim >= 0 : concatenate along this dimension
#       - None     : replicated across TP ranks (average)

#     You MUST customize this to match your actual param names.
#     Heuristics below are reasonable for LLaMA-like Megatron configs.
#     """
#     # 1D params (LayerNorm, some biases) are typically replicated.
#     if tensor.ndim == 1:
#         return None

#     # Embeddings / vocab-parallel
#     if "word_embeddings" in name or "tok_embeddings" in name:
#         return 0

#     # Attention QKV / MLP up-projection (column-parallel → shard output dim 0)
#     if "query_key_value" in name or "qkv" in name:
#         return 0
#     if "dense_h_to_4h" in name or "up_proj" in name or "gate_proj" in name:
#         return 0

#     # MLP down-projection / attention output projection (row-parallel → shard input dim 1)
#     if "dense_4h_to_h" in name or "down_proj" in name or "o_proj" in name:
#         return 1

#     # Final LM head often vocab-parallel
#     if "output_layer.weight" in name or "lm_head.weight" in name:
#         return 0

#     # Fallback: force you to explicitly handle new names
#     raise ValueError(f"Don't know shard dim for parameter {name} with shape {tuple(tensor.shape)}")


###############################################################################
# Core merge logic
###############################################################################

def merge_model_states(state0, state1):
    """
    Merge two TP rank model state dicts into a single TP=1 state dict.
    Returns OrderedDict(name -> merged_tensor) with same key order as state0.
    """
    names0, shapes0, _, _ = compute_slices(state0)
    names1, shapes1, _, _ = compute_slices(state1)
    assert names0 == names1, "Parameter name mismatch between TP ranks"
    merged = OrderedDict()

    for name, shape0, shape1 in zip(names0, shapes0, shapes1):
        t0 = state0[name]
        t1 = state1[name]
        assert t0.shape == shape0 and t1.shape == shape1

        shard_dim = get_shard_dim(name, t0)
        if shard_dim is None:
            merged_tensor = 0.5 * (t0 + t1)
        else:
            merged_tensor = torch.cat([t0, t1], dim=shard_dim)

        merged[name] = merged_tensor

    return merged


def merge_optimizer_flats(state0, state1,
                          flat_p0, flat_m0, flat_v0,
                          flat_p1, flat_m1, flat_v1):
    """
    Using the same param ordering as state0/state1, slice flat optimizer buffers
    for each TP rank and merge them into per-parameter tensors.
    Returns dicts name->tensor for param, exp_avg, exp_avg_sq.
    """
    names0, shapes0, numels0, offsets0 = compute_slices(state0)
    names1, shapes1, numels1, offsets1 = compute_slices(state1)
    assert names0 == names1, "Parameter name mismatch between TP ranks"
    total0 = sum(numels0)
    total1 = sum(numels1)

    assert flat_p0.numel() == total0 == flat_m0.numel() == flat_v0.numel(), \
        "Rank 0 flat optimizer size mismatch"
    assert flat_p1.numel() == total1 == flat_m1.numel() == flat_v1.numel(), \
        "Rank 1 flat optimizer size mismatch"

    merged_param = OrderedDict()
    merged_exp_avg = OrderedDict()
    merged_exp_avg_sq = OrderedDict()

    for name, shape0, n0, off0, shape1, n1, off1 in zip(
        names0, shapes0, numels0, offsets0, shapes1, numels1, offsets1
    ):
        t0 = state0[name]
        t1 = state1[name]
        assert t0.shape == shape0 and t1.shape == shape1

        p0 = flat_p0[off0:off0 + n0].view(shape0)
        m0 = flat_m0[off0:off0 + n0].view(shape0)
        v0 = flat_v0[off0:off0 + n0].view(shape0)

        p1 = flat_p1[off1:off1 + n1].view(shape1)
        m1 = flat_m1[off1:off1 + n1].view(shape1)
        v1 = flat_v1[off1:off1 + n1].view(shape1)

        shard_dim = get_shard_dim(name, t0)
        if shard_dim is None:
            mp = 0.5 * (p0 + p1)
            mm = 0.5 * (m0 + m1)
            mv = 0.5 * (v0 + v1)
        else:
            mp = torch.cat([p0, p1], dim=shard_dim)
            mm = torch.cat([m0, m1], dim=shard_dim)
            mv = torch.cat([v0, v1], dim=shard_dim)

        merged_param[name] = mp
        merged_exp_avg[name] = mm
        merged_exp_avg_sq[name] = mv

    return merged_param, merged_exp_avg, merged_exp_avg_sq


def flatten_merged_optimizer(merged_param, merged_exp_avg, merged_exp_avg_sq):
    """
    Given merged per-param tensors (ordered dicts), flatten them back into
    contiguous 1D tensors in the same param order.
    """
    names = list(merged_param.keys())
    flat_p = torch.cat([merged_param[n].reshape(-1) for n in names], dim=0)
    flat_m = torch.cat([merged_exp_avg[n].reshape(-1) for n in names], dim=0)
    flat_v = torch.cat([merged_exp_avg_sq[n].reshape(-1) for n in names], dim=0)
    return flat_p, flat_m, flat_v


###############################################################################
# Main script
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Merge Megatron TP=2 checkpoint (model + distrib_optim) into TP=1."
    )
    parser.add_argument(
        "--iter-dir",
        required=True,
        help="Path to iteration dir, e.g. .../checkpoints/iter_0720000",
    )
    parser.add_argument(
        "--out-dir",
        default="_tp1",
        help="Suffix for new merged iteration dir (default: _tp1).",
    )
    parser.add_argument(
        "--group-idx",
        type=int,
        default=0,
        help="Optimizer param group index to merge (default: 0).",
    )
    parser.add_argument(
        "--use-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Which mixed-precision group to merge (key in distrib_optim.pt).",
    )
    args = parser.parse_args()

    iter_dir = args.iter_dir
    out_iter_dir = iter_dir + args.out_dir

    os.makedirs(os.path.join(out_iter_dir, "mp_rank_00"), exist_ok=True)

    # Pick dtype pair for the optimizer group
    if args.use_dtype == "bf16":
        dtype_pair = (torch.bfloat16, torch.float32)
    else:
        dtype_pair = (torch.float16, torch.float32)

    # Paths
    model0_path = os.path.join(iter_dir, "mp_rank_00", "model_optim_rng.pt")
    model1_path = os.path.join(iter_dir, "mp_rank_01", "model_optim_rng.pt")
    optim0_path = os.path.join(iter_dir, "mp_rank_00", "distrib_optim.pt")
    optim1_path = os.path.join(iter_dir, "mp_rank_01", "distrib_optim.pt")

    # 1) Load model states
    ckpt0, state0 = load_model_state(model0_path)
    ckpt1, state1 = load_model_state(model1_path)

    # 2) Merge model tensors (TP=1)
    merged_model_state = merge_model_states(state0, state1)

    # 3) Build new model_optim_rng.pt (rank 0, TP=1) from ckpt0 as template
    new_model_ckpt = copy.deepcopy(ckpt0)
    new_model_ckpt["model"] = merged_model_state
    # Optional: if args has tensor_model_parallel_size, set to 1
    if hasattr(new_model_ckpt["args"], "tensor_model_parallel_size"):
        new_model_ckpt["args"].tensor_model_parallel_size = 1

    out_model_path = os.path.join(out_iter_dir, "mp_rank_00", "model_optim_rng.pt")
    torch.save(new_model_ckpt, out_model_path)
    print(f"Saved merged model checkpoint to: {out_model_path}")

    # 4) Load distrib_optim.pt for both ranks
    opt0, group0, flat_p0, flat_m0, flat_v0 = load_flat_optim(
        optim0_path, group_idx=args.group_idx, dtype_pair=dtype_pair
    )
    opt1, group1, flat_p1, flat_m1, flat_v1 = load_flat_optim(
        optim1_path, group_idx=args.group_idx, dtype_pair=dtype_pair
    )

    # 5) Merge optimizer states at per-param level
    merged_param, merged_exp_avg, merged_exp_avg_sq = merge_optimizer_flats(
        state0, state1,
        flat_p0, flat_m0, flat_v0,
        flat_p1, flat_m1, flat_v1,
    )

    # 6) Flatten merged optimizer back to single 1D buffers
    merged_flat_p, merged_flat_m, merged_flat_v = flatten_merged_optimizer(
        merged_param, merged_exp_avg, merged_exp_avg_sq
    )

    # 7) Build new distrib_optim.pt using opt0 as template
    new_opt = copy.deepcopy(opt0)
    new_group = new_opt[args.group_idx][dtype_pair]
    new_group["param"] = merged_flat_p
    new_group["exp_avg"] = merged_flat_m
    new_group["exp_avg_sq"] = merged_flat_v

    out_opt_path = os.path.join(out_iter_dir, "mp_rank_00", "distrib_optim.pt")
    torch.save(new_opt, out_opt_path)
    print(f"Saved merged optimizer checkpoint to: {out_opt_path}")

    print("Done. Use this new iteration dir with tensor_model_parallel_size=1.")


if __name__ == "__main__":
    main()