"""데코레이터 성능 오버헤드 측정 (과제 5.2-D: "decorator 성능 오버헤드 분석").

@timed / @logged / @validate_path 래퍼가 호출당 더하는 비용을 정량화한다.
순수 호출 오버헤드만 보기 위해 본문이 거의 없는 함수를 다회 호출하여
(데코레이터 적용 - 미적용) 차이를 호출당 나노초로 환산한다.

결과: results/decorator_overhead.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.after.data_pipeline.decorators import logged, timed, validate_path  # noqa: E402

N = 2_000_000
REPEATS = 5


def _bare(x):
    return x


@timed
def _t(x):
    return x


@logged()
@timed
def _tl(x):
    return x


@validate_path(suffixes=(".jsonl",), arg_index=0)
def _v(path, x):
    return x


def _time_call(fn, *args):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(N):
            fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best  # seconds for N calls (best-of)


def main():
    base = _time_call(_bare, 1)
    cases = [
        ("@timed", _time_call(_t, 1)),
        ("@logged+@timed", _time_call(_tl, 1)),
        ("@validate_path", _time_call(_v, "a.jsonl", 1)),
    ]
    rows = []
    base_ns = base / N * 1e9
    print(f"baseline (no decorator): {base_ns:.1f} ns/call")
    for name, t in cases:
        per_call_ns = t / N * 1e9
        overhead_ns = per_call_ns - base_ns
        rows.append({"decorator": name, "per_call_ns": round(per_call_ns, 1),
                     "overhead_ns": round(overhead_ns, 1), "n_calls": N})
        print(f"{name:>16}: {per_call_ns:7.1f} ns/call  (overhead {overhead_ns:+.1f} ns)")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "decorator_overhead.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decorator", "per_call_ns", "overhead_ns", "n_calls"])
        w.writeheader()
        w.writerows([{"decorator": "baseline", "per_call_ns": round(base_ns, 1),
                      "overhead_ns": 0.0, "n_calls": N}] + rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
