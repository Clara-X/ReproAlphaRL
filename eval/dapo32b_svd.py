import argparse
import gc
import os

import torch
from transformers import AutoModelForCausalLM


def parse_int_list(arg):
    if not arg:
        return None
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def iter_weight_params(module):
    for name, param in module.named_parameters():
        if name.endswith(".weight") and param.dim() == 2:
            yield name, param


def compute_topk_svd(diff, rank, oversample, niter):
    q = min(rank + oversample, min(diff.shape))
    u, s, v = torch.svd_lowrank(diff, q=q, niter=niter)
    order = torch.argsort(s, descending=True)[:rank]
    u = u[:, order].contiguous()
    s = s[order].contiguous()
    v = v[:, order].contiguous()
    vt = v.transpose(0, 1).contiguous()
    return u, s, vt


def compute_local_rank(rows, cols, rank, param_ratio):
    if param_ratio is None:
        return rank
    budget_rank = int((param_ratio * rows * cols) // (rows + cols + 1))
    return max(1, min(min(rows, cols), budget_rank))


def save_topk_svd_components(
    base_model_path,
    target_model_path,
    output_dir,
    rank=1,
    param_ratio=None,
    oversample=4,
    niter=4,
    layer_indices=None,
):
    print(f"🧩 Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    print(f"🧩 Loading target model from {target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    target_model.eval()

    selected_layers = set(layer_indices) if layer_indices else None
    svd_components = {
        "_meta": {
            "rank": rank,
            "param_ratio": param_ratio,
            "base_model_path": base_model_path,
            "target_model_path": target_model_path,
        }
    }

    for layer_idx, (base_layer, target_layer) in enumerate(zip(base_model.model.layers, target_model.model.layers)):
        if selected_layers is not None and layer_idx not in selected_layers:
            continue

        print(f"\n🔹 Computing top-{rank} SVD for layer {layer_idx}")
        layer_svd = {}

        for prefix, base_module, target_module in (
            ("self_attn", base_layer.self_attn, target_layer.self_attn),
            ("mlp", base_layer.mlp, target_layer.mlp),
        ):
            base_params = dict(iter_weight_params(base_module))
            for name, target_param in iter_weight_params(target_module):
                base_param = base_params[name]
                diff = (target_param.detach().to(torch.float32) - base_param.detach().to(torch.float32)).cpu()
                fro_norm = torch.linalg.matrix_norm(diff, ord="fro")
                local_rank = compute_local_rank(diff.shape[0], diff.shape[1], rank=rank, param_ratio=param_ratio)
                if fro_norm.item() == 0:
                    rows, cols = diff.shape
                    u = torch.zeros((rows, local_rank), dtype=torch.float32)
                    s = torch.zeros((local_rank,), dtype=torch.float32)
                    vt = torch.zeros((local_rank, cols), dtype=torch.float32)
                else:
                    u, s, vt = compute_topk_svd(diff, rank=local_rank, oversample=oversample, niter=niter)
                key_prefix = f"{prefix}_{name}"
                layer_svd[f"{key_prefix}_U"] = u.cpu()
                layer_svd[f"{key_prefix}_S"] = s.cpu()
                layer_svd[f"{key_prefix}_Vt"] = vt.cpu()
                layer_svd[f"{key_prefix}_fro_norm"] = fro_norm.cpu()
                print(
                    f"  [{prefix}] {name} | shape={tuple(target_param.shape)} | "
                    f"stored_rank={u.shape[1]} | fro_norm={fro_norm.item():.4f}"
                )
                del diff, u, s, vt, fro_norm
                gc.collect()

        svd_components[f"layer_{layer_idx}"] = layer_svd

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "svd_components.pt")
    torch.save(svd_components, save_path)
    print(f"\n✅ Saved compact SVD components to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a compact top-k SVD between a base model and a target model.")
    parser.add_argument("--base_model_path", required=True, help="Base model path.")
    parser.add_argument("--target_model_path", required=True, help="Target model path.")
    parser.add_argument("--output_dir", required=True, help="Directory to save svd_components.pt.")
    parser.add_argument("--rank", type=int, default=1, help="Number of singular components to keep.")
    parser.add_argument(
        "--param_ratio",
        type=float,
        default=None,
        help="Optional per-matrix low-rank parameter budget ratio. For example, 0.01 approximates Rank-1%%.",
    )
    parser.add_argument("--oversample", type=int, default=4, help="Oversampling value passed to torch.svd_lowrank.")
    parser.add_argument("--niter", type=int, default=4, help="Number of power iterations for torch.svd_lowrank.")
    parser.add_argument(
        "--layer_indices",
        default=None,
        help="Optional comma-separated layer indices for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_topk_svd_components(
        base_model_path=args.base_model_path,
        target_model_path=args.target_model_path,
        output_dir=args.output_dir,
        rank=args.rank,
        param_ratio=args.param_ratio,
        oversample=args.oversample,
        niter=args.niter,
        layer_indices=parse_int_list(args.layer_indices),
    )


if __name__ == "__main__":
    main()
