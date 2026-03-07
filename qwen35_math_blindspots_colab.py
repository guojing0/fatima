from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction


# Config
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MIN_MISTAKES = 10
DATASET_SIZE = 10
MAX_NEW_TOKENS = 80

# Greedy first, then sampled fallback if needed.
GENERATION_SETTINGS = [
    {"name": "greedy", "do_sample": False},
    {"name": "sample", "do_sample": True, "temperature": 0.8, "top_p": 0.9},
]

PUSH_TO_HUB = False
DATASET_REPO_ID = "YOUR_HF_USERNAME/qwen35_4b_base_math_blindspots"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

OUTPUT_DIR = "qwen35_math_output"
README_PATH = "README.md"
NUMBER_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+|-?\d+\.\d+|-?\d+")


PROBLEMS_RAW = """
p01|arithmetic|Compute 98765 * 4321.|426763565
p02|arithmetic|Compute (84 * 17) - (39 * 22).|570
p03|arithmetic|Compute (99999 + 88888) - (77777 + 66666).|44444
p04|fractions|Compute 3/7 + 5/14.|11/14
p05|fractions|Compute (7/9) * (27/14).|3/2
p06|fractions|Compute (5/6 - 1/4) / (1/3).|7/4
p07|decimals|Compute 0.125 * 64.|8
p08|decimals|Compute 1.2 + 3.45 - 0.78.|3.87
p09|percentages|18 is what percent of 72?|25
p10|percentages|Increase 250 by 12 percent.|280
p11|percentages|Decrease 480 by 15 percent.|408
p12|algebra|Solve for x: 7x - 15 = 34.|7
p13|algebra|Solve for x: 2x + 3 = 5x - 9.|4
p14|algebra|If x + y = 17 and x - y = 5, what is x?|11
p15|algebra|Solve for a: 3a + 2b = 18 and a - b = 2.|22/5
p16|exponents_logs|Solve for x: 2^(x+1) = 64.|5
p17|exponents_logs|Solve for x: log10(x) = 3.|1000
p18|modular_arithmetic|Compute 17^5 mod 23.|21
p19|modular_arithmetic|Compute 7^222 mod 13.|12
p20|number_theory|Find gcd(123456, 7890).|6
p21|number_theory|Find lcm(84, 126).|252
p22|combinatorics|Compute C(20, 6).|38760
p23|combinatorics|Compute 10P4.|5040
p24|probability|From a standard 52-card deck, what is the probability that two cards drawn without replacement are both aces?|1/221
p25|sequences|Compute the sum 1 + 2 + ... + 50.|1275
p26|sequences|Compute the sum of squares 1^2 + 2^2 + ... + 20^2.|2870
p27|geometry|A triangle has base 13 and height 9. What is its area?|117/2
p28|geometry|A rectangle has length 17 and width 9. What is its perimeter?|52
p29|geometry|A cube has side length 5. What is its total surface area?|150
p30|rates|A car travels 150 km in 2.5 hours. What is the average speed in km/h?|60
p31|rates|If 5 workers finish a job in 12 days, how many days for 8 workers at the same rate?|15/2
p32|ratios|If boys:girls = 3:5 and total students are 64, how many boys are there?|24
p33|calculus|If f(x) = x^3 - 4x^2 + 7x, compute f'(2).|3
p34|calculus|Compute the definite integral of 2x from 0 to 5.|25
p35|matrices|Find the determinant of [[1, 2], [3, 4]].|-2
p36|matrices|Let A=[[1,2],[0,1]] and B=[[3,1],[2,5]]. Find entry (1,2) of AB.|11
p37|word_problems|A number plus 9 equals twice the number minus 3. What is the number?|12
p38|statistics|What is the mean of the first 10 positive integers?|11/2
p39|statistics|Find the median of [12, 5, 8, 10, 9].|9
p40|statistics|Find the mode of [2, 3, 3, 4, 4, 4, 5].|4
p41|number_theory|Find the remainder when 1234567 is divided by 37.|25
p42|algebra|Solve for x: x/5 = 7/10.|7/2
p43|sequences|Arithmetic sequence with a1=4 and d=7. Find a20.|137
p44|sequences|Compute 3 + 6 + 12 + 24 + 48.|93
p45|percentages|Convert 37.5 percent to a fraction in simplest form.|3/8
p46|finance|Simple interest on principal 1200 at 5 percent per year for 3 years.|180
p47|finance|Compound amount for 1000 at 10 percent annual interest for 2 years.|1210
p48|angles|What is the smaller angle between clock hands at 3:30?|75
p49|counting|How many integers from 1 to 100 are divisible by 6?|16
p50|primes|What is the smallest prime greater than 100?|101
""".strip()


def install_if_missing() -> None:
    required = {
        "transformers": "transformers>=4.49.0",
        "datasets": "datasets>=2.19.0",
        "accelerate": "accelerate>=0.34.0",
    }
    missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        print("All required packages are already installed.")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    print("Installed:", ", ".join(missing))


def get_problem_bank() -> list[dict[str, str]]:
    rows = []
    for line in PROBLEMS_RAW.splitlines():
        pid, topic, prompt, expected = line.split("|", 3)
        rows.append({"id": pid, "topic": topic, "input": prompt, "expected_output": expected})
    return rows


def make_prompt(question: str) -> str:
    return (
        "You are solving a math problem.\n"
        "Return only the final numeric answer. No explanation.\n"
        f"Problem: {question}\n"
        "Final answer:"
    )


def parse_number(text: str | None) -> Fraction | None:
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


def extract_numeric_answer(text: str) -> str:
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


def answer_is_correct(expected_output: str, model_output: str) -> tuple[bool, str]:
    parsed_answer = extract_numeric_answer(model_output)
    expected_num = parse_number(expected_output)
    predicted_num = parse_number(parsed_answer)

    if expected_num is None or predicted_num is None:
        return parsed_answer.lower() == expected_output.strip().lower(), parsed_answer

    if predicted_num == expected_num:
        return True, parsed_answer

    try:
        return abs(float(predicted_num) - float(expected_num)) <= 1e-6, parsed_answer
    except Exception:
        return False, parsed_answer


def collect_mistakes(generator, tokenizer, problems: list[dict[str, str]]) -> list[dict[str, str]]:
    mistakes_by_id: dict[str, dict[str, str]] = {}

    for setting in GENERATION_SETTINGS:
        print(f"\nRunning mode: {setting['name']}")
        for idx, problem in enumerate(problems, start=1):
            if problem["id"] in mistakes_by_id:
                continue

            kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "return_full_text": True,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": setting["do_sample"],
            }
            if setting["do_sample"]:
                kwargs["temperature"] = setting["temperature"]
                kwargs["top_p"] = setting["top_p"]

            prompt = make_prompt(problem["input"])
            generated = generator(prompt, **kwargs)[0]["generated_text"]
            completion = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
            is_correct, parsed_answer = answer_is_correct(problem["expected_output"], completion)

            if not is_correct:
                mistakes_by_id[problem["id"]] = {
                    "id": problem["id"],
                    "topic": problem["topic"],
                    "input": problem["input"],
                    "expected_output": problem["expected_output"],
                    "model_output": completion,
                    "parsed_model_answer": parsed_answer,
                    "generation_mode": setting["name"],
                    "model_id": MODEL_ID,
                }

            if idx % 10 == 0:
                print(f"  Checked {idx}/{len(problems)} prompts - mistakes found: {len(mistakes_by_id)}")

        if len(mistakes_by_id) >= MIN_MISTAKES:
            break

    return list(mistakes_by_id.values())


def select_diverse_mistakes(rows: list[dict[str, str]], target_size: int) -> list[dict[str, str]]:
    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_topic[row["topic"]].append(row)

    topics = sorted(by_topic.keys(), key=lambda topic: len(by_topic[topic]), reverse=True)
    selected: list[dict[str, str]] = []

    for topic in topics:
        if by_topic[topic]:
            selected.append(by_topic[topic].pop(0))
        if len(selected) >= target_size:
            return selected[:target_size]

    while len(selected) < target_size:
        made_progress = False
        for topic in topics:
            if by_topic[topic]:
                selected.append(by_topic[topic].pop(0))
                made_progress = True
            if len(selected) >= target_size:
                break
        if not made_progress:
            break

    return selected


def save_dataset(selected_rows: list[dict[str, str]]) -> str:
    from datasets import Dataset, DatasetDict

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = Dataset.from_list(selected_rows)
    dataset_dict = DatasetDict({"train": dataset})

    dataset_path = os.path.join(OUTPUT_DIR, "dataset")
    dataset_dict.save_to_disk(dataset_path)

    print("Saved local output:")
    print("-", dataset_path)
    return dataset_path


def print_cli_upload_steps(dataset_path: str) -> None:
    if not PUSH_TO_HUB:
        print("PUSH_TO_HUB is False. Set it to True when you are ready to upload.")
        return

    if DATASET_REPO_ID.startswith("YOUR_HF_USERNAME/"):
        raise ValueError("Please set DATASET_REPO_ID before pushing.")

    print("\nRun these commands in Colab to upload with Hugging Face CLI:")
    print("pip install -q huggingface_hub")
    print("huggingface-cli login")
    print(f"huggingface-cli repo create {DATASET_REPO_ID} --type dataset")
    print(f'huggingface-cli upload {DATASET_REPO_ID} "{dataset_path}" . --repo-type dataset')
    print(f'huggingface-cli upload {DATASET_REPO_ID} "{README_PATH}" README.md --repo-type dataset')
    print(f"Target repo: https://huggingface.co/datasets/{DATASET_REPO_ID}")

    if HF_TOKEN:
        print("HF_TOKEN was found in environment. You can also login with: huggingface-cli login --token <token>")


def main() -> None:
    install_if_missing()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    problems = get_problem_bank()
    print(f"Loaded {len(problems)} math prompts.")

    if not torch.cuda.is_available():
        print("Warning: GPU is not available. Running a 4B model can be slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    all_mistakes = collect_mistakes(generator, tokenizer, problems)
    print(f"\nUnique mistaken prompts found: {len(all_mistakes)}")

    if len(all_mistakes) < MIN_MISTAKES:
        raise RuntimeError(
            f"Only found {len(all_mistakes)} mistakes. Increase prompt diversity or add another generation mode."
        )

    selected_rows = select_diverse_mistakes(all_mistakes, DATASET_SIZE)
    print(f"Selected {len(selected_rows)} rows for final dataset.")
    print("Topic distribution:", dict(Counter(row["topic"] for row in selected_rows)))

    for row in selected_rows[:5]:
        print("\n---")
        print("Topic:", row["topic"])
        print("Input:", row["input"])
        print("Expected:", row["expected_output"])
        print("Model output:", row["model_output"][:200])

    dataset_path = save_dataset(selected_rows)
    print_cli_upload_steps(dataset_path)


if __name__ == "__main__":
    main()
