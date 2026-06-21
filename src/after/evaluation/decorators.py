"""AFTER: 평가용 데코레이터 (카테고리 D).

- `timed`: 단계별 실행 시간 계측.
- `retry`: 불안정한 외부 생성 백엔드(vLLM OOM/일시적 실패) 호출에 지수 백오프 재시도.
  과제에서 명시한 "불안정 외부 API 호출에 retry decorator" 항목에 해당.
- 무거운 객체(harmony 인코딩 등)는 `functools.lru_cache` 로 1회만 생성.
모두 `functools.wraps` 로 metadata 보존, 예외를 삼키지 않는다(재시도 소진 시 마지막 예외 재발생).
"""

from __future__ import annotations

import functools
import time
from typing import Callable


def timed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.last_elapsed = time.perf_counter() - start
    wrapper.last_elapsed = 0.0
    return wrapper


def retry(max_attempts: int = 3, base_delay: float = 0.0,
          exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    """지수 백오프 재시도 데코레이터(parameterized).

    base_delay=0.0 이면 sleep 없이 즉시 재시도(테스트/벤치마크용).
    모든 시도 실패 시 마지막 예외를 그대로 raise(예외 비은닉).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt + 1 == max_attempts:
                        raise
                    if base_delay:
                        time.sleep(base_delay * (2 ** attempt))
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator


@functools.lru_cache(maxsize=None)
def get_harmony_encoding(name: str = "HARMONY_GPT_OSS"):
    """harmony 인코딩 로더 캐싱 자리.

    원본은 `_budget_forcing`/`convert_requests_harmony_template` 마다 재로딩했다.
    실제 환경에서는 `load_harmony_encoding(HarmonyEncodingName[name])` 를 1회만 호출하면 된다.
    벤치마크 환경에는 openai_harmony 가 없으므로 import 가능 여부만 다룬다.
    """
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding  # type: ignore
    return load_harmony_encoding(HarmonyEncodingName[name])
