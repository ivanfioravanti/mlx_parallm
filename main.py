import mlx_parallm
import importlib
importlib.reload(mlx_parallm)

import argparse
import random
import string

from mlx_parallm.utils import load, batch_generate, load_mmlu_pro_prompts


def build_letter_prompts(n: int) -> list[str]:
    capital_letters = string.ascii_uppercase
    distinct_pairs = [
        (a, b) for i, a in enumerate(capital_letters) for b in capital_letters[i + 1 :]
    ]
    prompt_template = (
        "Think of a real word containing both the letters {l1} and {l2}. "
        "Then, say 3 sentences which use the word."
    )
    return [
        prompt_template.format(l1=p[0], l2=p[1])
        for p in random.sample(distinct_pairs, n)
    ]


def main():
    parser = argparse.ArgumentParser(description="MLX ParaLLM demo")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        help="HF repo or local path",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts to run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. If set, groups prompts by similar token length",
    )
    parser.add_argument(
        "--mmlu-pro",
        action="store_true",
        help="Use TIGER-Lab/MMLU-Pro prompts instead of toy prompts",
    )
    parser.add_argument(
        "--mmlu-split", type=str, default="test", help="MMLU-Pro split to use"
    )
    parser.add_argument(
        "--mmlu-seed",
        type=int,
        default=0,
        help="Seed for sampling MMLU-Pro prompts",
    )
    parser.add_argument(
        "--benchmark", 
        dest="benchmark",
        action="store_true",
        help="Benchmark mode: suppress text output and show only final stats",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help=">1.0 to discourage repeating tokens (applies in greedy and sampling)",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=64,
        help="How many recent tokens to consider for repetition penalty",
    )
    args = parser.parse_args()

    # load model
    model, tokenizer = load(args.model)

    if args.mmlu_pro:
        prompts = load_mmlu_pro_prompts(
            num_prompts=args.num_prompts, split=args.mmlu_split, seed=args.mmlu_seed
        )
    else:
        prompts = build_letter_prompts(args.num_prompts)

    # Prepare aggregation for benchmark mode
    agg_stats = {"prompt_time": 0.0, "gen_time": 0.0, "prompt_tokens": 0, "gen_tokens": 0, "prompts": 0, "batches": 0} if args.benchmark else None

    # If batch_size specified, group prompts by similar token length
    if args.batch_size and args.batch_size > 0 and args.batch_size < len(prompts):
        # Compute tokenized lengths using the same chat template formatting
        chats = [[{"role": "user", "content": p}] for p in prompts]
        formatted = [
            tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in chats
        ]
        tok = tokenizer._tokenizer(formatted)
        lengths = [len(ids) for ids in tok["input_ids"]]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])

        # Run batches of similar lengths
        for i in range(0, len(order), args.batch_size):
            idxs = order[i : i + args.batch_size]
            group = [prompts[j] for j in idxs]
            _ = batch_generate(
                model,
                tokenizer,
                prompts=group,
                max_tokens=100,
                verbose=not args.benchmark,
                show_text=not args.benchmark,
                stats=agg_stats,
                run_label=args.model,
                temp=0.0,
                repetition_penalty=args.repetition_penalty,
                repetition_context_size=args.repetition_context_size,
            )
    else:
        _ = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=100,
            verbose=not args.benchmark,
            show_text=not args.benchmark,
            stats=agg_stats,
            run_label=args.model,
            temp=0.0,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
        )

    # Print one final summary in benchmark mode
    if args.benchmark and agg_stats is not None:
        total_prompt_time = agg_stats["prompt_time"]
        total_gen_time = agg_stats["gen_time"]
        total_prompt_tokens = agg_stats["prompt_tokens"]
        total_gen_tokens = agg_stats["gen_tokens"]
        total_prompts = max(1, agg_stats["prompts"])  # avoid div by zero
        total_elapsed = total_prompt_time + total_gen_time
        prompt_tps = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0.0
        gen_tps = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0
        avg_elapsed = total_elapsed / total_prompts

        print("=" * 10)
        print(f"Model: {args.model}")
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        print(f"Total elapsed time: {total_elapsed:.3f}s")
        print(f"Average total elapsed time per prompt: {avg_elapsed:.3f}s")


if __name__ == "__main__":
    main()
