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
pretty_name: Qwen3.5-4B-Base Math Blind Spots
---

# Qwen3.5-4B-Base Math Blind Spots

This dataset contains math prompts where `Qwen/Qwen3.5-4B-Base` produced incorrect answers.

- Model tested: [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- Fields: `id`, `input`, `expected_output`, `model_output`, `parsed_model_answer`, `model_id`

## How The Model Was Loaded

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "Qwen/Qwen3.5-4B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

## Typical Blind Spots

Common failure types seen while collecting these examples:
- Multi-step arithmetic slips
- Fraction and percentage normalization mistakes
- Modular arithmetic errors
- Wrong algebra manipulation
- Format-following failures when asked for final answer only

## What To Fine-Tune On

To reduce these errors, fine-tune on a supervised math dataset with:
- Broad topic coverage: arithmetic, algebra, geometry, probability, number theory, rates, word problems
- Multiple paraphrases per problem type
- Mixed difficulty levels
- Strict normalized targets for integer, fraction, decimal, and percentage outputs

### How To Assemble It

A practical approach:
1. Start from open math datasets such as GSM8K-style problems, MATH-style symbolic tasks, and arithmetic-focused corpora.
2. Add synthetic template-generated problems for weak areas (fractions, modular arithmetic, percentages, unit/rate conversions).
3. Validate numeric answers automatically with programmatic checkers.
4. Keep a held-out blind-spot split for final evaluation.

### Suggested Size

- Minimum useful range: 30k to 60k high-quality supervised examples
- Stronger range: 100k to 300k examples with balanced topic distribution and filtering
