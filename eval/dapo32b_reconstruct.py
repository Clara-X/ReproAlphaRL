import argparse
import gc
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def iter_weight_params(module):
    for name, param in module.named_parameters():
        if name.endswith(".weight") and param.dim() == 2:
            yield name, param


def apply_rank_k_update(param, layer_svd, key_prefix, rank, alpha):
    key_u = f"{key_prefix}_U"
    if key_u not in layer_svd:
        return False

    u = layer_svd[key_u].to(torch.float32)
    s = layer_svd[f"{key_prefix}_S"].to(torch.float32)
    vt = layer_svd[f"{key_prefix}_Vt"].to(torch.float32)
    fro_norm = layer_svd[f"{key_prefix}_fro_norm"].to(torch.float32)

    used_rank = s.shape[0] if rank is None or rank <= 0 else min(rank, s.shape[0])
    u_k = u[:, :used_rank]
    s_k = s[:used_rank]
    vt_k = vt[:used_rank, :]

    update_k = u_k @ torch.diag(s_k) @ vt_k
    topk_norm = torch.linalg.matrix_norm(update_k, ord="fro")
    scale = 1.0
    if topk_norm.item() > 0:
        scale = (fro_norm / topk_norm).item()

    param.data.add_(alpha * scale * update_k.to(dtype=param.data.dtype))
    print(
        f"  [{key_prefix}] used_rank={used_rank} "
        f"topk_norm={topk_norm.item():.4f} full_norm={fro_norm.item():.4f} scale={scale:.4f}"
    )
    return True


def resolve_torch_dtype(config):
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        attr = dtype.replace("torch.", "")
        if hasattr(torch, attr):
            return getattr(torch, attr)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.bfloat16


def persist_tokenizer_fix(model_dir):
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    tokenizer_config["fix_mistral_regex"] = True
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"✅ Persisted fix_mistral_regex=True in {config_path}")


def reconstruct_rank_k(base_model_path, svd_path, output_dir, rank=-1, alpha=1.0):
    print(f"🧩 Loading base config and tokenizer from {base_model_path}")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    dtype = resolve_torch_dtype(config)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    tokenizer.init_kwargs["fix_mistral_regex"] = True
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🧩 Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    print(f"🧩 Loading compact SVD components from {svd_path}")
    svd_components = torch.load(svd_path, map_location="cpu")

    for layer_idx, layer in enumerate(base_model.model.layers):
        layer_svd = svd_components.get(f"layer_{layer_idx}", {})
        if not layer_svd:
            continue

        print(f"\n🔹 Reconstructing layer {layer_idx}")
        for prefix, module in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for name, param in iter_weight_params(module):
                apply_rank_k_update(
                    param=param,
                    layer_svd=layer_svd,
                    key_prefix=f"{prefix}_{name}",
                    rank=rank,
                    alpha=alpha,
                )

        gc.collect()

    base_model.config.torch_dtype = dtype
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        base_model.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.eos_token_id = tokenizer.eos_token_id

    os.makedirs(output_dir, exist_ok=True)
    base_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    persist_tokenizer_fix(output_dir)
    print(f"\n✅ Saved reconstructed rank-{rank} model to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct a rank-k DAPO* model from compact SVD components.")
    parser.add_argument("--base_model_path", required=True, help="Base model path.")
    parser.add_argument("--svd_path", required=True, help="Path to svd_components.pt.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the reconstructed model.")
    parser.add_argument("--rank", type=int, default=-1, help="Number of singular components to use. <=0 means use everything stored in svd_components.pt.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale factor applied after norm matching.")
    return parser.parse_args()


def main():
    args = parse_args()
    reconstruct_rank_k(
        base_model_path=args.base_model_path,
        svd_path=args.svd_path,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
