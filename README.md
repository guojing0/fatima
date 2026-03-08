---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- qwen3.5
- math
- evaluation
pretty_name: Qwen3.5-0.8B Math Blind Spots
---

[**Corresponding Google Colab Notebook**]()

# Qwen3.5-0.8B Math Blind Spots

This project uses `Qwen/Qwen3.5-0.8B` to solve math problems from different fields, and then saves the first 20 mistakes.

- Model tested: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- Script: [Code on Github]() or see [Colab notebook]()
- Output file format: `train.parquet`
- Fields: `id`, `input`, `expected_output`, `model_output`, `parsed_model_answer`

## Local run with uv

```bash
cd fatima
uv sync
uv run python qwen35-0.8b-math.py
```

It writes the output (first 20 mistakes) to `qwen35_math_output/train.parquet`.

## Upload the public dataset

```bash
uv run hf auth login
uv run hf repos create hedgehog0/Qwen3.5-0.8B-Math-Questions --repo-type dataset --private=false --exist-ok
uv run hf upload hedgehog0/Qwen3.5-0.8B-Math-Questions qwen35_math_output/train.parquet train.parquet --repo-type dataset
uv run hf upload hedgehog0/Qwen3.5-0.8B-Math-Questions README.md README.md --repo-type dataset
```

## How to load the model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    trust_remote_code=True,
)
if device != "cpu":
    model = model.to(device)
```

## Blind spots

Since I am experimenting with a small model (0.8B), I expect it to make some mistakes and I have noticed that it would make the following mistakes:

1. There are calculation slips when the computation is multi-steps.
2. It makes mistakes when converting fractions to percentages, and vice versa.
3. It has troubles with modular arithmetic.
4. It does not always do algebra calculation and manipulation correctly.
5. It sometimes fails to follow the format of the final answer, even if we ask explicitly.

## Fine-tuning datasets

**Questions**: Discuss what kind of dataset do you think the model should be fine-tuned on to fix such errors. How would you assemble or find such a dataset? How big of a dataset do you think you’d need?

**Answer**: To reduce these kinds of errors, we can use mixed curriculum datasets from popular public math datasets on Hugging Face, such as the following popular ones:

- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - strong baseline for grade-school arithmetic and word problems.
- [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) - large synthetic math QA data for instruction tuning.
- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) - chain-of-thought style math supervision from recent open training pipelines.
- [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) - modern open math instruct mixture with broad topic coverage.

For concrete steps, we could start with easier data (e.g., `gsm8k`, easier subset of `MetaMathQA`) to stabilize the answer format. Further, we can add medium and hard problems (e.g., `MathInstruct` and `NuminaMath-CoT`) for reasoning depth. To make sure that for answer format is correct, we only need to keep strict target normalization for integers, fractions, decimals, and percentages in our training pipeline.
