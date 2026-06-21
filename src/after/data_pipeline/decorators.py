"""AFTER: 데이터 파이프라인용 데코레이터 (카테고리 D).

핵심 연구 로직과 부가 실행 정책(계측·로깅·검증)을 분리한다.
모든 데코레이터는 `functools.wraps` 로 원본 함수의 metadata(__name__/__doc__/signature)를
보존하며, 예외를 숨기지 않는다.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Callable


def timed(func: Callable) -> Callable:
    """함수 실행 시간을 측정해 `func.last_elapsed` 에 기록하는 데코레이터."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.last_elapsed = time.perf_counter() - start
    wrapper.last_elapsed = 0.0
    return wrapper


def logged(level: str = "INFO") -> Callable:
    """parameterized 데코레이터 — 호출/완료를 로깅(기본은 조용함, verbose 시 출력)."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get("PIPELINE_VERBOSE"):
                print(f"[{level}] -> {func.__name__}")
            result = func(*args, **kwargs)
            if os.environ.get("PIPELINE_VERBOSE"):
                print(f"[{level}] <- {func.__name__}")
            return result
        return wrapper
    return decorator


def validate_path(*, must_exist: bool = False, suffixes: tuple[str, ...] | None = None,
                  arg_index: int = 0, arg_name: str | None = None) -> Callable:
    """경로 인자를 검증하는 데코레이터.

    원본 `save_jsonl` 의 인라인 `assert` 를 대체한다. signature 를 망가뜨리지 않고
    검증 실패 시 명확한 예외를 던지며(예외 비은닉) 원본 함수를 그대로 호출한다.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name is not None and arg_name in kwargs:
                path = kwargs[arg_name]
            else:
                path = args[arg_index]
            if suffixes is not None and not str(path).endswith(suffixes):
                raise ValueError(f"path must end with one of {suffixes}: {path!r}")
            if must_exist and not os.path.exists(path):
                raise FileNotFoundError(f"path does not exist: {path!r}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
