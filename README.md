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
pretty_name: Qwen3.5-0.8B-Base Math Blind Spots
---

[**Corresponding Google Colab Notebook**](https://colab.research.google.com/drive/1AFqtgDWLMsdgXC-ka8kQL3bojfmMv0kf?usp=sharing)

# Qwen3.5-0.8B-Base Math Blind Spots

This project uses `Qwen/Qwen3.5-0.8B-Base` to solve math problems from different fields, and then saves the first 20 mistakes.

- Model tested: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (created on *February 28, 2026*)
- Script: [Code on Github](https://github.com/guojing0/fatima) or see [Colab notebook](https://colab.research.google.com/drive/1AFqtgDWLMsdgXC-ka8kQL3bojfmMv0kf?usp=sharing)
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

model_id = "Qwen/Qwen3.5-0.8B-Base"
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

### Mistake diversity summary (20 rows in `train.parquet`)

| Topic | Count | IDs | Typical failure |
|---|---:|---|---|
| Arithmetic and numeric sums | 3 | p01, p08, p26 | Arithmetic slip in multi-step numeric computation |
| Percent, ratio, rate, and probability word problems | 4 | p09, p24, p31, p32 | Mis-translation from words to the right formula |
| Algebra and equations | 4 | p15, p17, p42, p43 | Returns intermediate value instead of target variable |
| Number theory and divisibility | 4 | p18, p19, p20, p41 | Incorrect modular or gcd/remainder reasoning |
| Linear algebra | 2 | p35, p36 | Matrix operation mistakes |
| Statistics | 2 | p39, p40 | Picks wrong statistic (median/mode confusion) |
| Calculus | 1 | p33 | Derivative evaluation error |

## Fine-tuning datasets

**Questions**: Discuss what kind of dataset do you think the model should be fine-tuned on to fix such errors. How would you assemble or find such a dataset? How big of a dataset do you think you’d need?

**Answer**: To reduce these errors, I would assemble a mixed math SFT dataset that is balanced by topic and answer format.

Popular public sources:

- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - strong baseline for grade-school arithmetic and word problems.
- [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) - large synthetic math QA data for instruction tuning.
- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) - chain-of-thought style math supervision from recent open training pipelines.
- [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) - modern open math instruct mixture with broad topic coverage.

How I would assemble it:

1. Build a union of these datasets and keep only problem-answer pairs with parseable final answers.
2. Normalize targets into canonical forms for integer, fraction, decimal, and percent.
3. Add metadata labels (`topic`, `difficulty`, `answer_type`) and rebalance to match observed blind spots.
4. Deduplicate near-identical questions and filter low-quality or contradictory samples.
5. Keep a held-out evaluation split focused on the same blind-spot categories.

Recommended size:

- Minimum useful: 40k to 80k high-quality examples for measurable gains.
- Strong target: 150k to 300k balanced examples for robust improvement across all listed topics.
