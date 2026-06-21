"""Evaluator 미세 최적화 격리 벤치마크 (FakeBackend/GPU 불필요, CPU 전용).

5.2 self-consistency 시간에 합산돼 따로 보이지 않던 미세 최적화를 개별 격리해 정량화한다.
세 가지를 측정한다.

  (A) 집계(voting)  : statistics.mode(+예외) vs collections.Counter.most_common(1)
                      둘 다 O(n)이며 mode 는 내부적으로 Counter 를 쓰므로 차이는 상수배
                      (함수 래퍼·예외 경로 제거). 동률은 둘 다 first-seen 으로 결정적.
  (B) 반복본 생성    : extend([req]*n) vs itertools.chain/repeat. 최종 N×r 리스트는 양쪽
                      모두 materialize 하므로 peak memory 는 비슷하고, 차이는 질문마다
                      만들던 전이 임시 리스트([req]*n) 제거에서 오는 시간 상수배.
  (C) 재시도 복원력   : 불안정 백엔드(호출이 확률 p 로 예외)에서 @retry 미적용 vs 적용.
                      before 성공률 ≈ 1-p, after(@retry k회) ≈ 1-p^k.

각 항목은 before/after 출력 동치를 assert 로 확인한 뒤 measure(warm-up 1 + 7회)로 측정한다.
결과: results/micro_aggregate.csv, results/micro_repeat.csv, results/micro_retry.csv
"""

from __future__ import annotations

import csv
import os
import random
import sys
from statistics import StatisticsError, mode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from benchmark._bench import measure  # noqa: E402
from benchmark.make_synthetic_responses import build_eval_messages  # noqa: E402
from src.after.evaluation.aggregator import AnswerAggregator  # noqa: E402
from src.after.evaluation.decorators import retry  # noqa: E402
from src.after.evaluation.request_builder import RequestBuilder  # noqa: E402

REPEATS = 7
SEED = 1234


# ===== (A) 집계(voting): statistics.mode(+예외) vs Counter.most_common =====
def _before_majority(filtered: list[str]) -> str:
    """src/before/evaluation/evaluator_core.py:71-74 의 voting 을 그대로 재현."""
    try:
        return mode([i for i in filtered if i != "invalid"])
    except StatisticsError:
        return "invalid"


def _make_vote_groups(n_groups: int, repeats: int, seed: int = SEED):
    rng = random.Random(seed)
    letters = ["A", "B", "C", "D", "E", "invalid"]
    return [[rng.choice(letters) for _ in range(repeats)] for _ in range(n_groups)]


def bench_aggregate(n_groups: int, repeats: int) -> dict:
    groups = _make_vote_groups(n_groups, repeats)
    agg = AnswerAggregator(repeats)
    # 동치 검증: before voting == after voting (동률 처리 포함)
    fb = [_before_majority(g) for g in groups]
    fa = [agg._majority(g) for g in groups]
    assert fb == fa, f"voting mismatch n={n_groups} r={repeats}"

    mb = measure(lambda: [_before_majority(g) for g in groups], REPEATS)
    ma = measure(lambda: [agg._majority(g) for g in groups], REPEATS)
    return {"n": n_groups, "repeats": repeats,
            "before_mean_s": mb["mean_s"], "before_std_s": mb["std_s"],
            "after_mean_s": ma["mean_s"], "after_std_s": ma["std_s"],
            "speedup": mb["mean_s"] / ma["mean_s"] if ma["mean_s"] else 0.0}


# ===== (B) 반복본 생성: extend([req]*n) vs itertools.chain/repeat =====
def _before_repeat(requests: list, n: int) -> list:
    """src/before/evaluation/evaluator_core.py:42-45 의 반복본 생성을 그대로 재현."""
    repeated = list()
    for req in requests:
        repeated.extend([req] * n)
    return repeated


def bench_repeat(n_questions: int, repeats: int) -> dict:
    msgs, _ = build_eval_messages(n_questions, seed=SEED)
    rb = RequestBuilder()
    assert _before_repeat(msgs, repeats) == rb.repeat_for_self_consistency(msgs, repeats)

    mb = measure(lambda: _before_repeat(msgs, repeats), REPEATS)
    ma = measure(lambda: rb.repeat_for_self_consistency(msgs, repeats), REPEATS)
    return {"n": n_questions, "repeats": repeats,
            "before_mean_s": mb["mean_s"], "before_std_s": mb["std_s"], "before_peak_kib": mb["peak_kib"],
            "after_mean_s": ma["mean_s"], "after_std_s": ma["std_s"], "after_peak_kib": ma["peak_kib"],
            "speedup": mb["mean_s"] / ma["mean_s"] if ma["mean_s"] else 0.0,
            "mem_ratio": mb["peak_kib"] / ma["peak_kib"] if ma["peak_kib"] else 0.0}


# ===== (C) 재시도 복원력: @retry 미적용 vs 적용 =====
class _FlakyBackend:
    """generate 호출이 확률 p 로 예외를 던지는 불안정 백엔드(재현용 seed RNG)."""

    def __init__(self, p: float, seed: int):
        self.p = p
        self.rng = random.Random(seed)

    def generate(self, prompt: str) -> str:
        if self.rng.random() < self.p:
            raise RuntimeError("transient backend failure")
        return "ok"


def bench_retry(p: float, trials: int, max_attempts: int = 3, seed: int = SEED) -> dict:
    # before: 재시도 없음 → 첫 호출이 실패하면 그대로 실패
    be = _FlakyBackend(p, seed)
    before_ok = 0
    for _ in range(trials):
        try:
            be.generate("x")
            before_ok += 1
        except RuntimeError:
            pass

    # after: 동일 호출을 @retry(max_attempts) 로 감쌈. 평균 시도 횟수도 집계
    ae = _FlakyBackend(p, seed)
    attempts = {"n": 0}

    def _call():
        attempts["n"] += 1
        return ae.generate("x")

    wrapped = retry(max_attempts=max_attempts, base_delay=0.0)(_call)
    after_ok = 0
    for _ in range(trials):
        try:
            wrapped()
            after_ok += 1
        except RuntimeError:
            pass

    return {"p": p, "trials": trials, "max_attempts": max_attempts,
            "before_success_rate": round(before_ok / trials, 4),
            "after_success_rate": round(after_ok / trials, 4),
            "after_attempts_mean": round(attempts["n"] / trials, 3),
            "theory_after": round(1 - p ** max_attempts, 4)}


def _write(name: str, rows: list[dict]):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def main():
    # (A) 집계: 질문 수 N 을 키워 상수배 차이 확인 (반복 r=9 고정)
    agg_rows = []
    for n in [20000, 50000, 100000, 200000]:
        r = bench_aggregate(n, 9)
        agg_rows.append(r)
        print(f"[aggregate] N={n:>7} r=9: {r['before_mean_s']*1000:8.2f}ms -> "
              f"{r['after_mean_s']*1000:8.2f}ms ({r['speedup']:.2f}x)")
    _write("micro_aggregate.csv", agg_rows)

    # (B) 반복본 생성: 질문 수 N 변화 (반복 r=9 고정)
    rep_rows = []
    for n in [20000, 50000, 100000]:
        r = bench_repeat(n, 9)
        rep_rows.append(r)
        print(f"[repeat   ] N={n:>7} r=9: {r['before_mean_s']*1000:8.2f}ms -> "
              f"{r['after_mean_s']*1000:8.2f}ms ({r['speedup']:.2f}x), "
              f"mem {r['before_peak_kib']/1024:.1f} -> {r['after_peak_kib']/1024:.1f} MiB")
    _write("micro_repeat.csv", rep_rows)

    # (C) 재시도 복원력: 실패율 p 변화 (max_attempts=3)
    ret_rows = []
    for p in [0.1, 0.3, 0.5]:
        r = bench_retry(p, trials=20000, max_attempts=3)
        ret_rows.append(r)
        print(f"[retry    ] p={p}: before {r['before_success_rate']:.3f} -> "
              f"after {r['after_success_rate']:.3f} (theory {r['theory_after']:.3f}, "
              f"avg {r['after_attempts_mean']} attempts)")
    _write("micro_retry.csv", ret_rows)
    _plot(agg_rows, rep_rows, ret_rows)


def _plot(agg_rows, rep_rows, ret_rows):
    """3-패널 요약 그림: (좌) 집계·(중) 반복본은 before/after 거의 동일, (우) 재시도만 성공률 향상."""
    figdir = os.path.join(ROOT, "results", "figures")
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))

    def _bars(ax, rows, before_lab, after_lab, title):
        ns = [r["n"] for r in rows]
        x = list(range(len(ns)))
        before = [r["before_mean_s"] * 1000 for r in rows]
        after = [r["after_mean_s"] * 1000 for r in rows]
        w = 0.38
        ax.bar([i - w / 2 for i in x], before, w, label=before_lab, color="#c0504d")
        ax.bar([i + w / 2 for i in x], after, w, label=after_lab, color="#4f81bd")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n // 1000}k" for n in ns])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("N (questions)")
        ax.set_ylabel("time (ms)")
        ax.legend(fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.3)

    _bars(axes[0], agg_rows, "before (mode)", "after (Counter)", "(a) aggregate: mode vs Counter")
    _bars(axes[1], rep_rows, "before (extend)", "after (itertools)", "(b) repeat: extend vs itertools")

    ax = axes[2]
    ps = [r["p"] for r in ret_rows]
    ax.plot(ps, [r["before_success_rate"] for r in ret_rows], "o--", color="#c0504d", label="before (no retry)")
    ax.plot(ps, [r["after_success_rate"] for r in ret_rows], "s-", color="#4f81bd", label="after (@retry x3)")
    ax.plot(ps, [r["theory_after"] for r in ret_rows], "k:", label="theory  1 - p^3")
    ax.set_title("(c) retry resilience", fontsize=10)
    ax.set_xlabel("failure prob  p")
    ax.set_ylabel("success rate")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(figdir, "micro_compare.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
