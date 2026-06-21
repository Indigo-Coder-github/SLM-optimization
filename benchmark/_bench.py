"""공통 측정 유틸: 반복 실행 시간(평균·표준편차)과 tracemalloc peak memory.

tracemalloc 은 Python 레벨 할당을 추적한다(파이프라인 비교에서 중간 list 적재 vs
generator streaming 차이를 정확히 포착). pandas 의 C 버퍼는 포착되지 않으므로,
CSV 경로의 메모리 비교는 before 에 불리한 양을 오히려 *과소* 추정하는 보수적 값이다.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from typing import Callable


def measure(work: Callable[[], object], repeats: int = 7, warmup: int = 1) -> dict:
    """work() 를 반복 실행해 시간 통계와 peak memory 를 측정한다."""
    for _ in range(warmup):
        work()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        work()
        times.append(time.perf_counter() - t0)

    tracemalloc.start()
    work()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mean_s": statistics.fmean(times),
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "peak_kib": peak / 1024.0,
        "repeats": repeats,
    }
