import re
import sys
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_MODEL_FAMILIES = {"qwen3", "llama3", "llama31", "bloomz"}
PROMPT_VERSION = "llm_utils_v1"


def _normalize_model_family(model_family: str) -> str:
    family = (model_family or "qwen3").strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family={model_family!r}. Expected one of {sorted(SUPPORTED_MODEL_FAMILIES)}.")
    return family


def _resolve_torch_dtype(torch_dtype: Any, device: str) -> Any:
    if torch_dtype != "auto":
        if isinstance(torch_dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return dtype_map.get(torch_dtype.lower(), torch_dtype)
        return torch_dtype
    if device == "auto":
        return "auto"
    return torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    model_family: str,
    torch_dtype: Any = "auto",
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    _normalize_model_family(model_family)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(torch_dtype, device),
    }
    if device == "auto":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _build_qwen3_translation_prompt(
    src_lang: str,
    tgt_lang: str,
    query_src: str,
    demos: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are a professional machine translation system.",
        f"Translate from {src_lang} to {tgt_lang}.",
        f"Output only the translation in {tgt_lang}. Do not repeat words or phrases. Do not explain.",
        "Use the following examples as guidance.",
        "",
    ]

    for idx, demo in enumerate(demos, start=1):
        lines.extend(
            [
                f"Example {idx}:",
                f"{src_lang}: {demo.get('src', '')}",
                f"{tgt_lang}: {demo.get('tgt', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "Now translate the following sentence.",
            f"{src_lang}: {query_src}",
            f"{tgt_lang}:",
        ]
    )
    return "\n".join(lines)


def _build_llama31_user_content(
    src_lang: str,
    tgt_lang: str,
    query_src: str,
    demos: List[Dict[str, Any]],
) -> str:
    lines = [
        f"Translate from {src_lang} to {tgt_lang}.",
        f"Return only the {tgt_lang} translation.",
        "",
    ]
    if demos:
        lines.extend(["Examples:"])
        for demo in demos:
            lines.extend(
                [
                    f"{src_lang}: {demo.get('src', '')}",
                    f"{tgt_lang}: {demo.get('tgt', '')}",
                    "",
                ]
            )
    lines.extend(
        [
            "Now translate:",
            f"{src_lang}: {query_src}",
            f"{tgt_lang}:",
        ]
    )
    return "\n".join(lines)


def _apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def _build_bloomz_translation_prompt(
    src_lang: str,
    tgt_lang: str,
    query_src: str,
    demos: List[Dict[str, Any]],
) -> str:
    lines = [
        f"Task: Translate {src_lang} into {tgt_lang}. Output only the {tgt_lang} translation.",
        "",
    ]
    for demo in demos:
        lines.extend(
            [
                f"{src_lang}: {demo.get('src', '')}",
                f"{tgt_lang}: {demo.get('tgt', '')}",
                "",
            ]
        )
    lines.extend(
        [
            f"{src_lang}: {query_src}",
            f"{tgt_lang}:",
        ]
    )
    return "\n".join(lines)


def build_translation_prompt(
    src_lang: str,
    tgt_lang: str,
    query_src: str,
    demos: List[Dict[str, Any]],
    model_family: str,
    tokenizer: Any = None,
) -> str:
    family = _normalize_model_family(model_family)
    demos = demos or []
    if family == "qwen3":
        return _build_qwen3_translation_prompt(src_lang, tgt_lang, query_src, demos)
    if family in {"llama3", "llama31"}:
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate accurately. Return only the translation, with no explanations.",
            },
            {
                "role": "user",
                "content": _build_llama31_user_content(src_lang, tgt_lang, query_src, demos),
            },
        ]
        return _apply_chat_template(tokenizer, messages)
    return _build_bloomz_translation_prompt(src_lang, tgt_lang, query_src, demos)


def _format_qwen3_for_model(prompt: str, tokenizer: Any) -> str:
    user_content = prompt.strip() + "\n/no_think"
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_content


def _move_inputs_to_model_device(inputs: Dict[str, Any], model: Any) -> Dict[str, Any]:
    if hasattr(model, "hf_device_map"):
        return {key: value.to(model.device) if hasattr(model, "device") else value for key, value in inputs.items()}
    model_device = next(model.parameters()).device
    return {key: value.to(model_device) for key, value in inputs.items()}


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    model_family: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    family = _normalize_model_family(model_family)
    model_prompt = _format_qwen3_for_model(prompt, tokenizer) if family == "qwen3" else prompt
    inputs = tokenizer(model_prompt, return_tensors="pt")
    inputs = _move_inputs_to_model_device(inputs, model)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if family in {"llama3", "llama31"}:
        terminators = [tokenizer.eos_token_id]
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            terminators.append(eot_id)
        generation_kwargs["eos_token_id"] = terminators
    if do_sample and temperature is not None:
        generation_kwargs["temperature"] = temperature
    if family == "qwen3":
        generation_kwargs["repetition_penalty"] = 1.15
        generation_kwargs["no_repeat_ngram_size"] = 4

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    output_ids = outputs[0][input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def remove_thinking(text: str) -> str:
    text = text or ""
    if re.match(r"^\s*<think\b[^>]*>", text, flags=re.IGNORECASE) and not re.search(
        r"</think>", text, flags=re.IGNORECASE
    ):
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def warn_if_repeated_tokens(text: str, threshold: int = 10) -> None:
    previous = None
    count = 0
    for token in re.findall(r"\S+", text or ""):
        normalized = token.strip().lower()
        if normalized == previous:
            count += 1
        else:
            previous = normalized
            count = 1
        if normalized and count > threshold:
            print(
                f"Warning: token {token!r} repeated consecutively more than {threshold} times.",
                file=sys.stderr,
            )
            return


def detect_repetition(text: str, threshold: int = 10) -> Dict[str, Any]:
    previous = None
    count = 0
    max_count = 0
    max_token = ""
    for token in re.findall(r"\S+", text or ""):
        normalized = token.strip().lower()
        if normalized == previous:
            count += 1
        else:
            previous = normalized
            count = 1
        if normalized and count > max_count:
            max_count = count
            max_token = token
    return {
        "has_repetition": max_count > threshold,
        "repeated_token": max_token if max_count > threshold else "",
        "max_repetition_run": max_count,
    }


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _strip_output_prefixes(text: str, tgt_lang: str) -> str:
    prefixes = [
        rf"{re.escape(tgt_lang)}",
        "Translation",
        "Answer",
        "Output",
        "The translation is",
    ]
    prefix_pattern = r"^(?:" + "|".join(prefixes) + r")\s*[:\uFF1A]\s*"
    return re.sub(prefix_pattern, "", text.strip(), flags=re.IGNORECASE).strip()


def _strip_llama_assistant_tail(text: str) -> str:
    return re.split(r"(?:\.assistant\b|\n\s*assistant\b)", text or "", maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _strip_llama_leading_assistant(text: str) -> str:
    return re.sub(r"^\s*assistant\s*:?\s*", "", text or "", flags=re.IGNORECASE).strip()


def _clean_translation_candidate(text: str, tgt_lang: str, family: str) -> str:
    cleaned = text or ""
    if family in {"llama3", "llama31"}:
        cleaned = _strip_llama_leading_assistant(cleaned)
        cleaned = _strip_llama_assistant_tail(cleaned)
    cleaned = _strip_code_fence(cleaned)
    cleaned = _strip_output_prefixes(cleaned, tgt_lang)
    if family in {"llama3", "llama31"}:
        cleaned = _strip_llama_leading_assistant(cleaned)
        cleaned = _strip_llama_assistant_tail(cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r'^(["\']{2,}|["\'])', "", cleaned)
    cleaned = re.sub(r'(["\']{2,}|["\'])$', "", cleaned)
    return cleaned.strip()


def clean_translation_output(raw_output: str, tgt_lang: str, model_family: str) -> str:
    family = _normalize_model_family(model_family)
    cleaned = remove_thinking(raw_output)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    non_empty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidates = paragraphs or non_empty_lines or [cleaned.strip()]
    for candidate in candidates:
        cleaned = _clean_translation_candidate(candidate, tgt_lang, family)
        if cleaned:
            break
    else:
        cleaned = ""
    warn_if_repeated_tokens(cleaned)
    return cleaned
