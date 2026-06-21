"""Evaluator before/after 벤치마크 (FakeBackend, GPU 불필요).

두 시나리오:
  (1) self_consistency : test_time_scaling=False, num_repeats 변화.
      추출(extract_answer: list materialize vs search) + voting(mode vs Counter) 비교.
  (2) budget_forcing   : test_time_scaling=True, ignore_eos_num 고정, N 변화.
      `token_ids.pop(0)` O(n^2)  vs  `deque.popleft` O(n) 의 스케일링 차이.
before/after 가 동일한 final answer 시퀀스를 내는지 검증한다.

결과: results/evaluator_results.csv, results/figures/eval_*.png
"""

from __future__ import annotations

import csv
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from benchmark._bench import measure  # noqa: E402
from benchmark.make_synthetic_responses import FakeBackend, build_eval_messages  # noqa: E402
from src.before.evaluation.evaluator_core import Evaluator as BeforeEval  # noqa: E402
from src.after.evaluation.config import EvalConfig  # noqa: E402
from src.after.evaluation.evaluator import Evaluator as AfterEval  # noqa: E402

REPEATS = 7
SEED = 1234


def _run_before(ev, msgs, labels, **kw):
    random.seed(SEED)  # shuffle 경로 RNG 재현
    return ev.evaluate_model(msgs, labels, **kw)


def _run_after(ev, msgs, labels):
    random.seed(SEED)
    return ev.evaluate_model(msgs, labels)


def bench_self_consistency(n: int, num_repeats: int) -> dict:
    msgs, labels = build_eval_messages(n, seed=SEED)
    backend = FakeBackend()
    before = BeforeEval(backend, num_repeats=num_repeats)
    after = AfterEval(backend, EvalConfig(num_repeats=num_repeats))

    acc_b, fr_b = before.evaluate_model(msgs, labels)
    acc_a, fr_a = after.evaluate_model(msgs, labels)
    assert fr_b == fr_a, f"self_consistency mismatch n={n} r={num_repeats}"

    mb = measure(lambda: before.evaluate_model(msgs, labels), REPEATS)
    ma = measure(lambda: after.evaluate_model(msgs, labels), REPEATS)
    return {"scenario": "self_consistency", "n": n, "param": num_repeats, "before": mb, "after": ma}


def bench_budget_forcing(n: int, ignore_eos_num: int) -> dict:
    msgs, labels = build_eval_messages(n, seed=SEED)
    backend = FakeBackend()
    kw = dict(test_time_scaling=True, token_budget=100000, ignore_eos_num=ignore_eos_num)
    before = BeforeEval(backend, num_repeats=1)
    after = AfterEval(backend, EvalConfig(num_repeats=1, test_time_scaling=True,
                                          token_budget=100000, ignore_eos_num=ignore_eos_num))

    acc_b, fr_b = before.evaluate_model(msgs, labels, **kw)
    acc_a, fr_a = after.evaluate_model(msgs, labels)
    assert fr_b == fr_a, f"budget_forcing mismatch n={n} ieos={ignore_eos_num}"

    mb = measure(lambda: before.evaluate_model(msgs, labels, **kw), REPEATS)
    ma = measure(lambda: after.evaluate_model(msgs, labels), REPEATS)
    return {"scenario": "budget_forcing", "n": n, "param": ignore_eos_num, "before": mb, "after": ma}


def _row(r):
    mb, ma = r["before"], r["after"]
    return {
        "scenario": r["scenario"], "n": r["n"], "param": r["param"],
        "before_mean_s": mb["mean_s"], "before_std_s": mb["std_s"], "before_peak_kib": mb["peak_kib"],
        "after_mean_s": ma["mean_s"], "after_std_s": ma["std_s"], "after_peak_kib": ma["peak_kib"],
        "speedup": mb["mean_s"] / ma["mean_s"] if ma["mean_s"] else 0.0,
        "mem_ratio": mb["peak_kib"] / ma["peak_kib"] if ma["peak_kib"] else 0.0,
    }


def main():
    rows = []

    # 시나리오 1: self-consistency (extract + voting)
    for n in [500, 2000, 8000, 20000]:
        for r in [1, 5, 9]:
            res = bench_self_consistency(n, r)
            rows.append(_row(res))
            print(f"[self_consistency] n={n:>6} repeats={r}: "
                  f"{res['before']['mean_s']*1000:8.2f}ms -> {res['after']['mean_s']*1000:8.2f}ms "
                  f"({rows[-1]['speedup']:.2f}x)")

    # 시나리오 2: budget forcing (pop(0) O(n^2) vs deque O(n))
    for n in [2000, 6000, 12000, 20000, 30000]:
        ieos = 5
        res = bench_budget_forcing(n, ieos)
        rows.append(_row(res))
        print(f"[budget_forcing ] n={n:>6} ieos={ieos}: "
              f"{res['before']['mean_s']*1000:8.2f}ms -> {res['after']['mean_s']*1000:8.2f}ms "
              f"({rows[-1]['speedup']:.2f}x)")

    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    csv_path = os.path.join(ROOT, "results", "evaluator_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)
    _plot(rows)


def _plot(rows):
    figdir = os.path.join(ROOT, "results", "figures")

    # self-consistency: time vs N for each repeats (같은 r 은 같은 색, before=점선·after=실선)
    sc = [r for r in rows if r["scenario"] == "self_consistency"]
    plt.figure(figsize=(6, 4))
    for i, rep in enumerate(sorted({r["param"] for r in sc})):
        sub = [r for r in sc if r["param"] == rep]
        ns = [r["n"] for r in sub]
        color = f"C{i}"
        plt.plot(ns, [r["before_mean_s"] * 1000 for r in sub], marker="o", linestyle="--", color=color, label=f"before r={rep}")
        plt.plot(ns, [r["after_mean_s"] * 1000 for r in sub], marker="s", linestyle="-", color=color, label=f"after r={rep}")
    plt.xlabel("questions (N)"); plt.ylabel("time (ms)")
    plt.title("evaluator self-consistency: time vs N")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "eval_time_vs_n.png"), dpi=300)
    plt.close()

    # budget forcing scaling
    bf = [r for r in rows if r["scenario"] == "budget_forcing"]
    ns = [r["n"] for r in bf]
    plt.figure(figsize=(6, 4))
    plt.plot(ns, [r["before_mean_s"] * 1000 for r in bf], marker="o", label="before (pop(0), O(n^2))")
    plt.plot(ns, [r["after_mean_s"] * 1000 for r in bf], marker="s", label="after (deque, O(n))")
    plt.xlabel("questions (N)"); plt.ylabel("time (ms)")
    plt.title("budget forcing: pop(0) O(n^2) vs deque O(n)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "eval_budget_scaling.png"), dpi=300)
    plt.close()
    print("wrote figures to", figdir)


if __name__ == "__main__":
    main()
