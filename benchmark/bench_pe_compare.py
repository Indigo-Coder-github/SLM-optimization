"""실제 전처리·Evaluator 벤치마크를 P코어 고정 vs E코어 고정으로 실행·비교.

측정 CPU(i5-12600K)는 P코어(고성능)·E코어(고효율)가 섞인 Alder Lake 하이브리드다.
본 프로젝트의 벤치마크는 모두 단일 스레드라, 프로세스를 어느 코어 종류에 고정하느냐에 따라
같은 코드의 절대 실행시간이 달라진다. 이를 직접 확인하기 위해, run_benchmark_pipeline /
run_benchmark_evaluator 의 실제 벤치마크 함수를 그대로 재사용하되 프로세스를 P코어와 E코어에
각각 affinity 고정하여 동일 입력으로 실행한다.

결과: results/pe_compare.csv, results/figures/pe_compare.png
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import benchmark.run_benchmark_pipeline as PIPE  # noqa: E402
import benchmark.run_benchmark_evaluator as EVAL  # noqa: E402

# 비교용으로 반복 횟수 축소(코어 2종 × 시나리오 4개의 총 런타임 관리)
PIPE.REPEATS = 5
EVAL.REPEATS = 5

# 한글 폰트(그래프 라벨용)
_FONT = os.path.join(ROOT, "report", "fonts", "NanumGothic-Regular.ttf")
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

PIPE_N = 20000
EVAL_N = 8000


def _probe(cpu: int, proc: psutil.Process) -> float:
    proc.cpu_affinity([cpu])
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        s = 0
        for i in range(800_000):
            s += (i * i) % 7
        best = min(best, time.perf_counter() - t0)
    return best


def pick_p_e(proc: psutil.Process, n: int):
    """짧은 probe 로 가장 빠른 코어(P 대표)·가장 느린 코어(E 대표)를 고른다."""
    times = {c: _probe(c, proc) for c in range(n)}
    p_cpu = min(times, key=times.get)
    e_cpu = max(times, key=times.get)
    return p_cpu, e_cpu, times


def _row(scn, core, mb, ma):
    return {
        "scenario": scn, "core": core,
        "before_ms": round(mb["mean_s"] * 1000, 2), "before_std_ms": round(mb["std_s"] * 1000, 2),
        "after_ms": round(ma["mean_s"] * 1000, 2), "after_std_ms": round(ma["std_s"] * 1000, 2),
        "speedup": round(mb["mean_s"] / ma["mean_s"], 3) if ma["mean_s"] else 0.0,
    }


def run_for_core(label: str, cpu: int, proc: psutil.Process):
    proc.cpu_affinity([cpu])
    out = []

    wd = tempfile.mkdtemp(prefix="pe_csv_")
    try:
        r = PIPE.bench_supervised_csv(PIPE_N, wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    out.append(_row("전처리 CSV (N=20k)", label, r["before"], r["after"]))

    wd = tempfile.mkdtemp(prefix="pe_jsonl_")
    try:
        r = PIPE.bench_meerkat_jsonl(PIPE_N, wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    out.append(_row("전처리 JSONL (N=20k)", label, r["before"], r["after"]))

    r = EVAL.bench_self_consistency(EVAL_N, 5)
    out.append(_row("Eval 다수결 (N=8k,r=5)", label, r["before"], r["after"]))

    r = EVAL.bench_budget_forcing(EVAL_N, 4)
    out.append(_row("Eval budget (N=8k,ieos=4)", label, r["before"], r["after"]))
    return out


def main():
    n = os.cpu_count()
    proc = psutil.Process()
    original = proc.cpu_affinity()
    p_cpu, e_cpu, probe = pick_p_e(proc, n)
    print(f"[pick] P-core 대표 = CPU{p_cpu} ({probe[p_cpu]*1000:.0f}ms), "
          f"E-core 대표 = CPU{e_cpu} ({probe[e_cpu]*1000:.0f}ms), 비율 {probe[e_cpu]/probe[p_cpu]:.2f}x")

    rows = []
    try:
        print("[run] P-core 고정 ...")
        rows += run_for_core("P", p_cpu, proc)
        print("[run] E-core 고정 ...")
        rows += run_for_core("E", e_cpu, proc)
    finally:
        proc.cpu_affinity(original)

    for r in rows:
        print(f"  [{r['core']}] {r['scenario']:<24} before {r['before_ms']:8.1f}ms  "
              f"after {r['after_ms']:8.1f}ms  speedup {r['speedup']:.2f}x")

    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    csv_path = os.path.join(ROOT, "results", "pe_compare.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)
    _plot(rows)


def _plot(rows):
    figdir = os.path.join(ROOT, "results", "figures")
    scen = []
    for r in rows:
        if r["scenario"] not in scen:
            scen.append(r["scenario"])
    by = {(r["scenario"], r["core"]): r for r in rows}
    import numpy as np
    x = np.arange(len(scen))
    w = 0.38

    # (1) after 절대 실행시간: P vs E
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    p_after = [by[(s, "P")]["after_ms"] for s in scen]
    e_after = [by[(s, "E")]["after_ms"] for s in scen]
    ax1.bar(x - w / 2, p_after, w, label="P코어", color="#2c5aa0")
    ax1.bar(x + w / 2, e_after, w, label="E코어", color="#e08a1e")
    ax1.set_xticks(x); ax1.set_xticklabels(scen, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("after 실행시간 (ms)"); ax1.set_title("최적화 후 절대 실행시간: P vs E")
    ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)

    # (2) speedup(개선 배수): P vs E
    p_sp = [by[(s, "P")]["speedup"] for s in scen]
    e_sp = [by[(s, "E")]["speedup"] for s in scen]
    ax2.bar(x - w / 2, p_sp, w, label="P코어", color="#2c5aa0")
    ax2.bar(x + w / 2, e_sp, w, label="E코어", color="#e08a1e")
    ax2.set_xticks(x); ax2.set_xticklabels(scen, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("speedup (before/after)"); ax2.set_title("최적화 개선 배수: P vs E")
    ax2.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(figdir, "pe_compare.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
