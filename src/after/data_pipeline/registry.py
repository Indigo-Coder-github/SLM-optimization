"""AFTER: 변환 함수 Registry / Factory (카테고리 C — OCP).

원본은 행마다 `match folder_name` / `match filename` 으로 분기했다.
여기서는 (1) 분기를 루프 **밖에서 1회** 결정하고, (2) 새 데이터셋을 추가할 때
`@register(...)` 로 등록만 하면 되도록 개방-폐쇄 원칙을 적용한다.

`RowTransformer` Protocol 로 인터페이스를 명시한다.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from src.common.chat_templates import (
    kormedmcqa_and_aihub_chat_template,
    pubmedqa_chat_template,
)


@runtime_checkable
class RowTransformer(Protocol):
    """row(dict) 와 file_stem(str) 을 받아 chat messages 를 만드는 변환자."""
    def __call__(self, row: dict, file_stem: str) -> list[dict[str, str]]: ...


_REGISTRY: dict[str, RowTransformer] = {}


def register(*folder_names: str) -> Callable[[RowTransformer], RowTransformer]:
    """변환 함수를 폴더 이름들에 등록하는 데코레이터."""
    def deco(fn: RowTransformer) -> RowTransformer:
        for name in folder_names:
            _REGISTRY[name] = fn
        return fn
    return deco


def get_transformer(folder_name: str) -> RowTransformer:
    """등록된 변환자를 O(1) dict 조회로 1회만 가져온다."""
    try:
        return _REGISTRY[folder_name]
    except KeyError:
        raise ValueError(f"Unknown folder name: {folder_name}") from None


# --- 등록 -------------------------------------------------------------------
@register("KorMedMCQA_Gemini", "AI_Hub_Multi_Choice")
def _kormedmcqa(row: dict, file_stem: str) -> list[dict[str, str]]:
    return kormedmcqa_and_aihub_chat_template(row, file_stem)


@register("PubMedQA")
def _pubmedqa(row: dict, file_stem: str) -> list[dict[str, str]]:
    return pubmedqa_chat_template(row)
