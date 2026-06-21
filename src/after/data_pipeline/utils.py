"""AFTER: 최적화된 utils.

핵심 변경
- `extract_answer_from_response`: 정규식 패턴 객체를 **모듈 레벨에서 1회 생성·재사용**한다.
  (원본은 호출마다 패턴 *문자열* 로 `re.finditer` 를 부르므로, CPython 의 내부 패턴 캐시(`re._cache`)
   조회 비용 — 캐시 미스 시 재컴파일 — 이 매 호출 발생한다. 미리 컴파일한 객체를 쓰면 그 조회 비용까지 없앤다.)
  또한 모든 match 를 리스트로 만들지 않고 `finditer` 이터레이터를 **그대로 역순/순방향 소비**하여
  필요한 첫 match 만 찾는다(할당 O(matches) → O(1)).  → 카테고리 A
- `save_jsonl`: list 뿐 아니라 **임의의 iterable/generator** 를 받아 라인 단위로 스트리밍 기록.
  호출 측에서 중간 리스트를 만들 필요가 없어진다.  → 카테고리 B
- `iter_jsonl`: 파일을 한 줄씩 lazy 하게 내보내는 제너레이터(전체 적재 없음).  → 카테고리 B
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Iterator

from .decorators import validate_path

# 모듈 로드시 1회만 컴파일해 패턴 객체 재사용.
# (원본은 호출마다 패턴 문자열로 re.finditer 호출 → CPython 내부 캐시 조회 비용 발생)
_KEYWORD_RE = re.compile(
    r'정답|답변|따라서|Answer|answer|Therefore|Thus|Correct choice|Correct option|답:')
_LETTER_RE = re.compile(r'[A-E]')
_LETTER_DOT_RE = re.compile(r'[A-E]\.')
_VALID_LETTERS = frozenset("ABCDE")


def iter_jsonl(file_path: str) -> Iterator[dict]:
    """`.jsonl` 을 한 줄씩 파싱해 내보내는 lazy 제너레이터."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(file_path: str) -> list[dict]:
    """원본 호환용 eager 버전(내부적으로 제너레이터를 소비)."""
    return list(iter_jsonl(file_path))


@validate_path(suffixes=(".json", ".jsonl"), arg_index=1, arg_name="file_path")
def save_jsonl(data: Iterable, file_path: str) -> int:
    """iterable 을 `.jsonl` 로 **스트리밍** 저장하고 기록한 라인 수를 반환.

    list 를 강제하지 않으므로 제너레이터를 그대로 흘려보낼 수 있다(peak memory 절감).
    """
    count = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_answer_from_response(response_text: str) -> str:
    """응답에서 [A-E] 정답 추출 — 원본과 **동일한 결과**, 더 적은 할당.

    - 키워드 매치를 리스트로 만들지 않고, 가장 마지막 키워드의 `end` 위치만 추적.
    - 마지막 키워드부터 거꾸로 내려가며 첫 `[A-E]` 를 `search` 로 한 번에 탐색.
    """
    last_keyword_ends: list[int] = []
    for m in _KEYWORD_RE.finditer(response_text):
        last_keyword_ends.append(m.end())

    if last_keyword_ends:
        # 마지막 키워드부터 역순으로 보며 그 뒤 첫 [A-E] 를 찾음(원본 의미 보존)
        for end in reversed(last_keyword_ends):
            m = _LETTER_RE.search(response_text, end)
            if m:
                return m.group()
    else:
        # prefix 가 없으면 마지막 "X." 패턴의 글자
        last = None
        for last in _LETTER_DOT_RE.finditer(response_text):
            pass
        if last is not None:
            return last.group()[0]

    return "invalid"


def is_valid_letter(ch: str) -> bool:
    """[A-E] 멤버십 테스트 — list 대신 frozenset 으로 O(1)."""
    return ch in _VALID_LETTERS
