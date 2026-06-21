"""Evaluator 벤치마크용 synthetic 요청/응답 생성기 + FakeBackend.

GPU/vLLM 없이 evaluator 의 host-side orchestration 을 측정하기 위해
vLLM `LLM` 과 동일한 표면(chat/generate/get_tokenizer)을 갖는 결정적 FakeBackend 를 제공한다.
입력 문자열이 같으면 항상 같은 출력을 내므로 before/after 가 동일 결과를 산출한다.
"""

from __future__ import annotations

import hashlib
import random


# --- 가짜 모델 출력 객체(vLLM RequestOutput 와 동일 표면) -------------------
class _Gen:
    __slots__ = ("text", "token_ids")

    def __init__(self, text: str, token_ids: list[int]):
        self.text = text
        self.token_ids = token_ids


class _Out:
    __slots__ = ("outputs",)

    def __init__(self, gen: _Gen):
        self.outputs = [gen]


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize: bool = False) -> str:
        if isinstance(messages, str):
            return messages
        return "\n".join(m["content"] for m in messages)


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _make_output(key_text: str) -> _Out:
    """key_text 로부터 결정적 응답을 생성.

    응답에는 여러 개의 흩어진 [A-E] 글자와 명시적 '정답: X' 를 포함하여
    extract_answer 의 리스트 materialize(before) 와 search(after) 차이를 측정 가능하게 한다.
    """
    h = _stable_hash(key_text)
    answer = "ABCDE"[h % 5]
    # 잡음 섞인 본문(여러 [A-E] 등장) + 마지막에 명시적 정답
    noise = " ".join(("A" if (h >> i) & 1 else "the") for i in range(20))
    text = f"Explanation: {noise} reasoning ... 정답: {answer}"
    token_ids = list(range((h % 40) + 8))
    return _Out(_Gen(text, token_ids))


class FakeBackend:
    """결정적 가짜 생성 백엔드 (GenerationBackend Protocol 충족)."""

    def __init__(self):
        self._tokenizer = _FakeTokenizer()

    def chat(self, requests: list[list[dict[str, str]]]) -> list[_Out]:
        return [_make_output(req[-1]["content"]) for req in requests]

    def generate(self, prompts: list[str]) -> list[_Out]:
        return [_make_output(p) for p in prompts]

    def get_tokenizer(self):
        return self._tokenizer


# --- 평가용 synthetic 요청(MCQA 형식 메시지) -------------------------------
_WORDS = "fever cough renal hepatic cardiac dose symptom lesion artery vein cell".split()


def _sentence(rng: random.Random, k=10) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(k))


def build_eval_messages(n: int, seed: int = 1234) -> tuple[list[list[dict[str, str]]], list[str]]:
    """n개의 4지선다 요청 메시지와 정답 레이블을 생성."""
    rng = random.Random(seed)
    messages, labels = [], []
    for _ in range(n):
        q = _sentence(rng, 12)
        opts = [_sentence(rng, 3) for _ in range(5)]
        user = (f"다음 질문에 답하세요.\n### 질문:\n{q}\n### 보기:\n"
                f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nE. {opts[4]}")
        messages.append([
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": user},
        ])
        labels.append(rng.choice("ABCDE"))
    return messages, labels


if __name__ == "__main__":
    msgs, labels = build_eval_messages(5)
    be = FakeBackend()
    outs = be.chat(msgs)
    print("sample fake output:", outs[0].outputs[0].text)
    print("num messages:", len(msgs), "labels:", labels)
