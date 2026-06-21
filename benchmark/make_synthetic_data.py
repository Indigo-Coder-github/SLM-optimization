"""전처리 벤치마크용 synthetic 데이터 생성기.

실제 연구 데이터(의료 QA)를 공개하지 않고도 동일한 구조를 재현하기 위해
seed 고정 난수로 CSV/JSONL 을 만든다. 모든 셀 값은 비어있지 않은 문자열로 생성하여
pandas(read_csv)와 csv.DictReader 가 동일한 dict 를 만들도록 보장한다(before↔after 출력 일치).
"""

from __future__ import annotations

import csv
import json
import os
import random

_WORDS = (
    "patient fever cough blood pressure renal hepatic cardiac dose mg symptom "
    "diagnosis therapy lesion artery vein cell tissue immune chronic acute "
    "환자 발열 기침 혈압 신장 간 심장 용량 증상 진단 치료 병변 동맥 정맥 세포 조직 면역 만성 급성"
).split()


def _sentence(rng: random.Random, k_min=6, k_max=18) -> str:
    n = rng.randint(k_min, k_max)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def write_supervised_csv(path: str, n: int, seed: int = 42) -> str:
    """instruction/output 2열 지도학습 CSV 생성 (SingleCsvProcessor 대상)."""
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instruction", "output"])
        for _ in range(n):
            w.writerow([_sentence(rng), _sentence(rng)])
    return path


def write_meerkat_multichoice_jsonl(path: str, n: int, seed: int = 42) -> str:
    """Meerkat 멀티초이스 JSONL 생성 (MultiJsonlSingleFolderProcessor 대상).

    user content 는 "(A)...(B)...(C)...(D)..." 형식, assistant 는
    "Therefore, the answer is (X)" 형식을 포함하도록 한다.
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    letters = "ABCD"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            question = _sentence(rng, 8, 20)
            opts = [_sentence(rng, 2, 5) for _ in range(4)]
            user = f"{question} (A) {opts[0]} (B) {opts[1]} (C) {opts[2]} (D) {opts[3]}"
            ans = rng.choice(letters)
            assistant = f"{_sentence(rng, 10, 25)} Therefore, the answer is ({ans})"
            row = {"messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    sample_dir = os.path.join(here, "..", "results", "_synthetic_sample")
    write_supervised_csv(os.path.join(sample_dir, "raw", "Sample-Healthinfo.csv"), 2000)
    write_meerkat_multichoice_jsonl(os.path.join(sample_dir, "raw", "Meerkat-Instructions", "MedQA-CoT.jsonl"), 2000)
    print("sample synthetic data written to", os.path.abspath(sample_dir))
