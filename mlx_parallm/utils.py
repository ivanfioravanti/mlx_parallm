# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
try:
    # Prefer public API
    from huggingface_hub.errors import RepositoryNotFoundError
except Exception:
    try:
        # Older versions exposed this in utils
        from huggingface_hub.utils import RepositoryNotFoundError
    except Exception:
        # Fallback to a local definition if not available
        class RepositoryNotFoundError(Exception):
            pass
from mlx.utils import tree_flatten
from transformers import PreTrainedTokenizer

# mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
# Note: tuner utils API changed in recent mlx_lm versions. Import lazily below.
# Keep dequantize import, which remains stable.
from mlx_lm.tuner.utils import dequantize as dequantize_model

# Local imports
from mlx_parallm.sample_utils import top_p_sampling
from mlx_parallm.models.base import BatchedKVCache

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def load_mmlu_pro_prompts(
    num_prompts: int,
    split: str = "test",
    subjects: Optional[List[str]] = None,
    seed: Optional[int] = None,
    instruction: Optional[str] = None,
) -> List[str]:
    """Load prompts from TIGER-Lab/MMLU-Pro and format them.

    This uses the Hugging Face `datasets` library. It samples `num_prompts`
    examples (shuffled deterministically with `seed` if provided) and returns
    formatted prompts suitable for `batch_generate`.

    Args:
        num_prompts: Number of prompts to sample.
        split: Dataset split to use, e.g. "test".
        subjects: Optional list of subject names to filter by.
        seed: Optional RNG seed for deterministic sampling.
        instruction: Optional instruction prefix to include before questions.

    Returns:
        List[str]: A list of formatted prompt strings.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "The 'datasets' package is required to load MMLU-Pro. "
            "Install it via `pip install datasets`."
        ) from e

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    if subjects:
        subjects_set = set(subjects)
        ds = ds.filter(lambda x: x.get("subject") in subjects_set)

    if seed is not None:
        ds = ds.shuffle(seed=seed)
    else:
        ds = ds.shuffle()

    take_n = min(num_prompts, len(ds))
    ds = ds.select(range(take_n))

    prompts: List[str] = []
    default_instruction = (
        "You are an expert at multiple-choice questions. "
        "Choose the single best answer."
    )
    prefix = instruction or default_instruction

    for ex in ds:
        question = (
            ex.get("question")
            or ex.get("query")
            or ex.get("prompt")
            or ""
        )
        options = ex.get("options") or ex.get("choices") or []

        if isinstance(options, dict):
            # Some datasets store choices as a dict mapping labels to text
            # Normalize to a list ordered by label where possible
            ordered = []
            for i in range(len(options)):
                label = chr(65 + i)
                if label in options:
                    ordered.append(options[label])
            if not ordered:  # fallback to values order
                ordered = list(options.values())
            options = ordered

        if options:
            opts_text = "\n".join(
                f"{chr(65 + i)}. {str(opt)}" for i, opt in enumerate(options)
            )
            prompt = (
                f"{prefix}\n"
                f"Question: {question}\n"
                f"Options:\n{opts_text}\n"
                f"Answer with only the letter (A, B, C, ...)."
            )
        else:
            prompt = f"{prefix}\nQuestion: {question}"

        prompts.append(prompt)

    return prompts


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    #return Model, ModelArgs
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_parallm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float) -> mx.array:
    """
    Apply a repetition penalty per batch row to discourage repeating tokens seen
    in the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits: [batch, vocab] logits for the next token.
        generated_tokens: Either a 1D sequence of token ids or a 2D array of
            shape [batch, seq] containing the recent context per batch row.
        penalty: Penalty factor (> 1.0 to discourage repetition).

    Returns:
        Logits with the penalty applied in-place and also returned.
    """
    # Fast path: nothing to do
    if generated_tokens is None:
        return logits

    # Helper to apply on a set of token indices for a specific row slice
    def _apply_row(row_slice: slice, tok_ids: list[int]):
        if not tok_ids:
            return
        unique_ids = list(set(int(t) for t in tok_ids))
        idx = mx.array(unique_ids)
        sel = logits[row_slice, idx]
        sel = mx.where(sel < 0, sel * penalty, sel / penalty)
        logits[row_slice, idx] = sel

    # Handle different container shapes for generated_tokens
    try:
        shape = getattr(generated_tokens, "shape", None)
        if shape is not None:
            # MX array path
            if len(shape) == 2:
                bsz = shape[0]
                for b in range(bsz):
                    _apply_row(slice(b, b + 1), generated_tokens[b].tolist())
            elif len(shape) == 1:
                # Same set for all rows
                _apply_row(slice(None), generated_tokens.tolist())
            else:
                # Unknown higher dims: flatten
                _apply_row(slice(None), mx.ravel(generated_tokens).tolist())
        else:
            # Python container
            if isinstance(generated_tokens[0], (list, tuple)):
                for b, row in enumerate(generated_tokens):
                    _apply_row(slice(b, b + 1), list(row))
            else:
                _apply_row(slice(None), list(generated_tokens))
    except Exception:
        # Fallback: best-effort on flattened content
        try:
            flat = list(generated_tokens)
        except Exception:
            return logits
        _apply_row(slice(None), flat)

    return logits


def generate_step(
    prompts: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        softmax_logits = mx.softmax(logits, axis=-1)

        if temp == 0:
            tokens = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            if top_p > 0 and top_p < 1.0:
                tokens = top_p_sampling(logits, top_p, temp)
            else:
                scaled_logits = logits * (1 / temp)
                tokens = mx.random.categorical(logits * (1 / temp), axis=-1)
                if scaled_logits.ndim > 1:
                    tokens = mx.expand_dims(tokens, axis=-1)

        probs = softmax_logits[0, tokens]
        return tokens, probs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    # (bs, ntoks)
    y = prompts
    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )

    cache = [BatchedKVCache(model.head_dim, n, y.shape[0]) for n in kv_heads]

    repetition_context = prompts

    if repetition_context_size and repetition_penalty:
        repetition_context = repetition_context[:,-repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        logits = model(y, cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(logits, repetition_context, repetition_penalty)
        y, probs = sample(logits)
        if repetition_penalty:
            repetition_context = mx.concatenate([repetition_context, y], axis=1)

        if repetition_context_size:
            if repetition_context.shape[1] > repetition_context_size:
                repetition_context = repetition_context[:,-repetition_context_size:]
        return y, probs

    y, p = _step(y)
    mx.async_eval(y)
    while True:
        next_y, next_p = _step(y)
        mx.async_eval(next_y)
        mx.eval(y)
        yield y, p
        y, p = next_y, next_p

def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    for (token, prob), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment

def batch_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = False,
    show_text: bool = True,
    format_prompts: bool = True,
    formatter: Optional[Callable] = None,
    stats: Optional[Dict[str, float]] = None,
    run_label: Optional[str] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print timing information and, if
           ``show_text`` is also ``True``, the prompts and generations.
           Default: ``False``.
       show_text (bool): When ``verbose=True``, controls whether to print the
           prompts and generated text. Default: ``True``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       stats (Optional[Dict[str, float]]): If provided, accumulates timing and
           token count metrics into this dictionary across calls. Keys used:
           'prompt_time', 'gen_time', 'prompt_tokens', 'gen_tokens',
           'prompts', 'batches'.
       run_label (Optional[str]): A label to print with metrics (e.g., model id).
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # verbose banner removed from the beginning; we'll print details after results

    if format_prompts:
        prompts_fm = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts_fm = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in prompts_fm]
    else:
        prompts_fm = prompts

    # left-padding for batched generation
    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts_toks = mx.array(tokenizer._tokenizer(prompts_fm, padding=True)['input_ids'])
    tic = time.perf_counter()

    output_toks = []
    for (tokens, _), n in zip(
        generate_step(prompts_toks, model, **kwargs),
        range(max_tokens),
    ): 
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        output_toks.append(tokens)
    output_toks = mx.concatenate(output_toks, axis=1)

    # detokenizing + stripping pad/eos tokens
    responses = [response.split(tokenizer.eos_token)[0].split(tokenizer.pad_token)[0] for response in tokenizer.batch_decode(output_toks.tolist())]
    # Compute metrics regardless of verbosity (for optional aggregation)
    gen_time = time.perf_counter() - tic
    prompt_tokens_count = int(prompts_toks.size)
    gen_tokens_count = int(output_toks.size)

    # Accumulate stats if requested
    if stats is not None:
        stats["prompt_time"] = stats.get("prompt_time", 0.0) + float(prompt_time)
        stats["gen_time"] = stats.get("gen_time", 0.0) + float(gen_time)
        stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + int(prompt_tokens_count)
        stats["gen_tokens"] = stats.get("gen_tokens", 0) + int(gen_tokens_count)
        stats["prompts"] = stats.get("prompts", 0) + int(len(prompts))
        stats["batches"] = stats.get("batches", 0) + 1

    if verbose:
        # Optionally print prompts and generations first
        if show_text:
            for prompt, response in zip(prompts, responses):
                print("=" * 10)
                print("Prompt:", prompt)
                print(response)
        # Then print speed metrics at the end
        prompt_tps = prompt_tokens_count / prompt_time
        gen_tps = gen_tokens_count / gen_time
        print("=" * 10)
        if run_label:
            print(f"Model: {run_label}")
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        total_elapsed = prompt_time + gen_time
        avg_elapsed = total_elapsed / max(1, len(prompts))
        print(f"Total elapsed time: {total_elapsed:.3f}s")
        print(f"Average total elapsed time per prompt: {avg_elapsed:.3f}s")
            
    return responses


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None]
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()

    for (token, prob), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                formatter(detokenizer.last_segment, prob.item())
            else:
                print(detokenizer.last_segment, end="", flush=True)

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        gen_time = time.perf_counter() - tic
        print(detokenizer.last_segment, flush=True)
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        print(f"Total elapsed time: {prompt_time + gen_time:.3f}s")

    return detokenizer.text


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config(dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle quantization configs which may include per-module overrides.
        # Expected base keys: group_size, bits. Optional overrides: path -> {group_size, bits}.
        base_group_size = quantization.get("group_size", 64)
        base_bits = quantization.get("bits", 4)

        # Build override map from quantization config for specific module paths
        override_map = {
            k: v
            for k, v in quantization.items()
            if isinstance(v, dict) and ("group_size" in v or "bits" in v)
        }

        # Only quantize modules that (a) support to_quantized and (b) have quantized tensors in weights
        def class_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            # Ensure corresponding quantized weights exist in files
            has_quant = f"{path}.scales" in weights
            if not has_quant:
                return False
            # Provide per-path overrides if present
            if path in override_map:
                ov = override_map[path]
                return {
                    "group_size": ov.get("group_size", base_group_size),
                    "bits": ov.get("bits", base_bits),
                }
            return True

        nn.quantize(
            model,
            group_size=base_group_size,
            bits=base_bits,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, model_config)
    if adapter_path is not None:
        # mlx_lm renamed apply_lora_layers -> load_adapters. Prefer new API,
        # fall back to the old name for backward compatibility.
        try:
            from mlx_lm.tuner.utils import load_adapters as _load_adapters  # type: ignore
        except Exception:
            try:
                from mlx_lm.tuner.utils import apply_lora_layers as _load_adapters  # type: ignore
            except Exception as e:
                raise ImportError(
                    "LoRA adapter loading API not found in mlx_lm. "
                    "Please upgrade mlx_lm or remove adapter_path."
                ) from e
        model = _load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    return model, config, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using mlx-lm version **{__version__}**.

        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")
        response = generate(model, tokenizer, prompt="hello", verbose=True)
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    weights = dict(tree_flatten(model.parameters()))
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits)

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)
