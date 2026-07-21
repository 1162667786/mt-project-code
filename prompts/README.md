# Prompt implementations

The exact MT-ICL prompt builders used in the experiments are implemented in
`src/llm_utils.py` under `PROMPT_VERSION = "llm_utils_v1"`.

- Qwen3 uses an instruction-style translation prompt, the model chat template,
  and `/no_think` handling.
- Llama 3 uses system and user messages through the tokenizer chat template.
- BLOOMZ uses a plain multilingual translation prompt.

Demonstrations are inserted in their retrieved order. The evaluation reference
is never inserted into a translation prompt. Use `--save_prompts` with
`scripts/run_mt_icl.py` to save the fully instantiated prompts for an execution.
