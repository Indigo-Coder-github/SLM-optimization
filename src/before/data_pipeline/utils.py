"""BEFORE: 원본 SLM `src/utils/utils.py` 를 그대로 옮긴 baseline.

알고리즘은 원본과 동일하게 유지한다(측정 공정성). 주석만 한국어로 보강.
"""

import json
import re


def load_jsonl(file_path: str) -> list[dict]:
    """`.jsonl` 파일을 읽어 한 줄씩 파싱한다."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return lines


def save_jsonl(data: list, file_path: str):
    """리스트를 `.jsonl` 로 저장한다."""
    assert isinstance(data, list), "data must be a list"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_answer_from_response(response_text: str) -> str:
    """응답에서 [A-E] 정답을 추출한다(원본 로직 그대로).

    매 호출마다 `re.finditer` 결과를 **리스트로 materialize** 하고,
    역순 순회하며 다시 `[A-E]` 매치를 전부 리스트로 만든다.
    """
    # 가능성 있는 prefix 키워드를 전부 수집(리스트 materialize)
    matches = [i for i in re.finditer(
        r'정답|답변|따라서|Answer|answer|Therefore|Thus|Correct choice|Correct option|답:',
        response_text)]

    if matches:
        # 가장 마지막 키워드 이후 텍스트에서 첫 [A-E] 를 찾음(역순 검색)
        for match in matches[::-1]:
            text_after_latest_keyword = response_text[match.end():]
            answer_matches = [i for i in re.finditer(r'[A-E]', text_after_latest_keyword)]
            if answer_matches:
                return answer_matches[0].group()
    else:
        # prefix 가 없으면 "A. ..." 형태로 가정
        matches = [i for i in re.finditer(r'[A-E]\.', response_text)]
        if matches:
            return matches[-1].group()[0]

    return "invalid"
