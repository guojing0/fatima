---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- math
- evaluation
- llm-blind-spots
pretty_name: Qwen3.5-0.8B Math Blind Spots
---

# Qwen3.5-0.8B Math Blind Spots

This project tests `Qwen/Qwen3.5-0.8B` on diverse math prompts and saves the first 10 mistakes.

- Model tested: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- Script: `qwen35-0.8b-math.py`
- Output file format: `train.parquet` (Hub dataset viewer friendly)
- Fields: `id`, `input`, `expected_output`, `model_output`, `parsed_model_answer`, `model_id`

## Local Run With uv

```bash
cd fatima
uv sync
uv run python qwen35-0.8b-math.py
```

The script writes:

- `qwen35_math_output/train.parquet`

## Upload As Public Dataset (hf CLI)

```bash
uv run hf auth login
uv run hf repos create hedgehog0/Qwen3.5-0.8B-Math-Questions --repo-type dataset --private=false --exist-ok
uv run hf upload hedgehog0/Qwen3.5-0.8B-Math-Questions qwen35_math_output/train.parquet train.parquet --repo-type dataset
uv run hf upload hedgehog0/Qwen3.5-0.8B-Math-Questions README.md README.md --repo-type dataset
```

## How The Model Is Loaded

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    trust_remote_code=True,
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

## Typical Blind Spots

Common failure types observed:

- Multi-step arithmetic slips
- Fraction and percentage normalization mistakes
- Modular arithmetic errors
- Wrong algebra manipulation
- Format-following failures when asked for final answer only

## Fine-Tuning Dataset Recommendation

To reduce these errors, fine-tune on a supervised math dataset with:

- Broad topic coverage: arithmetic, algebra, geometry, probability, number theory, rates, word problems
- Multiple paraphrases per problem type
- Mixed difficulty levels
- Strict normalized targets for integer, fraction, decimal, and percentage outputs

Suggested size:

- Minimum useful: 30k to 60k high-quality supervised examples
- Stronger: 100k to 300k examples with balanced topic coverage and filtering
