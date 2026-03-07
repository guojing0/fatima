import os
import re
from fractions import Fraction


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_SIZE = 10
MAX_NEW_TOKENS = 80

PUSH_TO_HUB = True
DATASET_REPO_ID = "hedgehog0/Qwen3.5-0.8B-Math-Questions"
OUTPUT_DIR = "qwen35_math_output"
PARQUET_NAME = "train.parquet"
README_PATH = "README.md"

NUMBER_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+|-?\d+\.\d+|-?\d+")

PROBLEMS = """
p01|Compute 98765 * 4321.|426763565
p02|Compute (84 * 17) - (39 * 22).|570
p03|Compute (99999 + 88888) - (77777 + 66666).|44444
p04|Compute 3/7 + 5/14.|11/14
p05|Compute (7/9) * (27/14).|3/2
p06|Compute (5/6 - 1/4) / (1/3).|7/4
p07|Compute 0.125 * 64.|8
p08|Compute 1.2 + 3.45 - 0.78.|3.87
p09|18 is what percent of 72?|25
p10|Increase 250 by 12 percent.|280
p11|Decrease 480 by 15 percent.|408
p12|Solve for x: 7x - 15 = 34.|7
p13|Solve for x: 2x + 3 = 5x - 9.|4
p14|If x + y = 17 and x - y = 5, what is x?|11
p15|Solve for a: 3a + 2b = 18 and a - b = 2.|22/5
p16|Solve for x: 2^(x+1) = 64.|5
p17|Solve for x: log10(x) = 3.|1000
p18|Compute 17^5 mod 23.|21
p19|Compute 7^222 mod 13.|12
p20|Find gcd(123456, 7890).|6
p21|Find lcm(84, 126).|252
p22|Compute C(20, 6).|38760
p23|Compute 10P4.|5040
p24|From a standard 52-card deck, what is the probability that two cards drawn without replacement are both aces?|1/221
p25|Compute the sum 1 + 2 + ... + 50.|1275
p26|Compute the sum of squares 1^2 + 2^2 + ... + 20^2.|2870
p27|A triangle has base 13 and height 9. What is its area?|117/2
p28|A rectangle has length 17 and width 9. What is its perimeter?|52
p29|A cube has side length 5. What is its total surface area?|150
p30|A car travels 150 km in 2.5 hours. What is the average speed in km/h?|60
p31|If 5 workers finish a job in 12 days, how many days for 8 workers at the same rate?|15/2
p32|If boys:girls = 3:5 and total students are 64, how many boys are there?|24
p33|If f(x) = x^3 - 4x^2 + 7x, compute f'(2).|3
p34|Compute the definite integral of 2x from 0 to 5.|25
p35|Find the determinant of [[1, 2], [3, 4]].|-2
p36|Let A=[[1,2],[0,1]] and B=[[3,1],[2,5]]. Find entry (1,2) of AB.|11
p37|A number plus 9 equals twice the number minus 3. What is the number?|12
p38|What is the mean of the first 10 positive integers?|11/2
p39|Find the median of [12, 5, 8, 10, 9].|9
p40|Find the mode of [2, 3, 3, 4, 4, 4, 5].|4
p41|Find the remainder when 1234567 is divided by 37.|25
p42|Solve for x: x/5 = 7/10.|7/2
p43|Arithmetic sequence with a1=4 and d=7. Find a20.|137
p44|Compute 3 + 6 + 12 + 24 + 48.|93
p45|Convert 37.5 percent to a fraction in simplest form.|3/8
p46|Simple interest on principal 1200 at 5 percent per year for 3 years.|180
p47|Compound amount for 1000 at 10 percent annual interest for 2 years.|1210
p48|What is the smaller angle between clock hands at 3:30?|75
p49|How many integers from 1 to 100 are divisible by 6?|16
p50|What is the smallest prime greater than 100?|101
""".strip()


def parse_number(text):
    if text is None:
        return None

    cleaned = text.strip().lower().replace(",", "").replace("−", "-").rstrip(".")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
        try:
            return Fraction(cleaned) / 100
        except Exception:
            return None

    try:
        return Fraction(cleaned)
    except Exception:
        return None


def extract_numeric_answer(text):
    if not text:
        return ""

    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[-1] if lines else text.strip()
    candidate = candidate.replace("Final answer:", "").replace("Answer:", "").strip()

    matches = NUMBER_PATTERN.findall(candidate)
    if not matches:
        matches = NUMBER_PATTERN.findall(text)

    return matches[-1].replace(" ", "") if matches else candidate


def is_correct(expected_output, model_output):
    parsed = extract_numeric_answer(model_output)
    expected_num = parse_number(expected_output)
    pred_num = parse_number(parsed)

    if expected_num is None or pred_num is None:
        return parsed.lower() == expected_output.strip().lower(), parsed
    if expected_num == pred_num:
        return True, parsed

    try:
        return abs(float(expected_num) - float(pred_num)) <= 1e-6, parsed
    except Exception:
        return False, parsed


def load_problems():
    rows = []
    for line in PROBLEMS.splitlines():
        pid, question, expected = line.split("|", 2)
        rows.append({"id": pid, "input": question, "expected_output": expected})
    return rows


def print_upload_commands(parquet_path):
    print("\nRun these commands to upload a public dataset with hf CLI:")
    print("uv run hf auth login")
    print(
        f"uv run hf repo create {DATASET_REPO_ID} --repo-type dataset --no-private --exist-ok"
    )
    print(
        f"uv run hf upload {DATASET_REPO_ID} \"{parquet_path}\" {PARQUET_NAME} --repo-type dataset"
    )
    print(
        f"uv run hf upload {DATASET_REPO_ID} \"{README_PATH}\" README.md --repo-type dataset"
    )


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    problems = load_problems()
    print(f"Loaded {len(problems)} math prompts.")
    print("Running mode: greedy")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    mistakes = []
    for idx, item in enumerate(problems, start=1):
        prompt = (
            "You are solving a math problem.\n"
            "Do not think. Do not show steps.\n"
            "Return only the final numeric answer.\n"
            f"Problem: {item['input']}\n"
            "Final answer:"
        )

        output = generator(
            prompt,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]

        completion = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
        correct, parsed = is_correct(item["expected_output"], completion)

        if not correct:
            mistakes.append(
                {
                    "id": item["id"],
                    "input": item["input"],
                    "expected_output": item["expected_output"],
                    "model_output": completion,
                    "parsed_model_answer": parsed,
                    "model_id": MODEL_ID,
                }
            )
            if len(mistakes) >= DATASET_SIZE:
                break

        if idx % 10 == 0:
            print(f"Checked {idx}/{len(problems)} prompts - mistakes found: {len(mistakes)}")

    print(f"\nMistakes collected: {len(mistakes)}")
    if len(mistakes) < DATASET_SIZE:
        raise RuntimeError(f"Only found {len(mistakes)} mistakes. Add more prompts.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = Dataset.from_list(mistakes)
    parquet_path = os.path.join(OUTPUT_DIR, PARQUET_NAME)
    dataset.to_parquet(parquet_path)

    print("Saved local output:")
    print("-", parquet_path)

    if PUSH_TO_HUB:
        print_upload_commands(parquet_path)


if __name__ == "__main__":
    main()
