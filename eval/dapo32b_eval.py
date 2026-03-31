import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BENCHMARKS = ("aime24", "minerva", "gpqa", "math", "gsm8k")
DEFAULT_MODEL_KEYS = ("full", "rank_1pct")


def parse_csv(arg):
    return [item.strip() for item in arg.split(",") if item.strip()]


def expected_output_path(output_dir, model_path, data_name, split, temperature, n_sampling):
    model_name = "/".join(Path(model_path).parts[-3:])
    out_prefix = f"{split}_t{temperature}"
    return os.path.join(output_dir, model_name, data_name, f"{out_prefix}_k{n_sampling}.jsonl")


def build_eval_command(python_bin, model_path, data_name, output_dir, args):
    return [
        python_bin,
        "-m",
        "reasoning_eval",
        "--model_name_or_path",
        model_path,
        "--data_name",
        data_name,
        "--temperature",
        str(args.temperature),
        "--start_idx",
        str(args.start_idx),
        "--end_idx",
        str(args.end_idx),
        "--n_sampling",
        str(args.n_sampling),
        "--k",
        str(args.k),
        "--split",
        args.split,
        "--max_tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_dir,
    ]


def parse_args():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run reasoning_eval.py against full and rank-1 DAPO-Qwen-32B models.")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used to launch reasoning_eval.")
    parser.add_argument(
        "--full_model_path",
        default=os.path.join(current_file_path, "models", "dapo32b", "full"),
        help="Path to the full DAPO-Qwen-32B model.",
    )
    parser.add_argument(
        "--rank1_model_path",
        default=os.path.join(current_file_path, "models", "dapo32b", "rank_1"),
        help="Path to the reconstructed legacy rank-1 model.",
    )
    parser.add_argument(
        "--rank1pct_model_path",
        default=os.path.join(current_file_path, "models", "dapo32b", "rank_1pct"),
        help="Path to the reconstructed rank-1%% model.",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark names.",
    )
    parser.add_argument(
        "--model_keys",
        default=",".join(DEFAULT_MODEL_KEYS),
        help="Comma-separated model keys from {full,rank_1}.",
    )
    parser.add_argument("--output_dir", default=os.path.join(current_file_path, "output", "dapo32b_runthrough"))
    parser.add_argument("--cuda_visible_devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n_sampling", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip model/benchmark pairs whose target output file already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    benchmarks = parse_csv(args.benchmarks)
    model_keys = parse_csv(args.model_keys)
    model_map = {
        "full": args.full_model_path,
        "rank_1": args.rank1_model_path,
        "rank_1pct": args.rank1pct_model_path,
    }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = f"{current_file_path}:{env.get('PYTHONPATH', '')}".rstrip(":")

    os.makedirs(args.output_dir, exist_ok=True)

    for model_key in model_keys:
        if model_key not in model_map:
            raise ValueError(f"Unsupported model key: {model_key}")
        model_path = model_map[model_key]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        for benchmark in benchmarks:
            out_file = expected_output_path(
                output_dir=args.output_dir,
                model_path=model_path,
                data_name=benchmark,
                split=args.split,
                temperature=args.temperature,
                n_sampling=args.n_sampling,
            )
            if args.skip_existing and os.path.exists(out_file):
                print(f"⏭️  Skip existing result: {out_file}")
                continue

            print(f"\n🚀 Evaluating {model_key} on {benchmark}")
            cmd = build_eval_command(
                python_bin=args.python_bin,
                model_path=model_path,
                data_name=benchmark,
                output_dir=args.output_dir,
                args=args,
            )
            subprocess.run(cmd, cwd=current_file_path, env=env, check=True)
            print(f"✅ Finished {model_key} on {benchmark}")


if __name__ == "__main__":
    main()
