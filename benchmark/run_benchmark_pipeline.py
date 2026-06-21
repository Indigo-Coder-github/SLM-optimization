"""전처리 파이프라인 before/after 벤치마크.

두 시나리오:
  (1) supervised_csv : pd.read_csv + iterrows  vs  csv.DictReader 스트리밍
  (2) meerkat_jsonl  : readlines 전체 적재     vs  라인 스트리밍 제너레이터
입력 크기 N 를 변화시키며 실행 시간(평균±표준편차)과 peak memory 를 측정하고,
before/after 출력 JSONL 의 해시가 동일한지 검증(정확성 회귀 없음)한다.

결과: results/pipeline_results.csv,
      results/figures/pipeline_time_*.png, pipeline_mem_*.png
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from benchmark._bench import measure  # noqa: E402
from benchmark.make_synthetic_data import (  # noqa: E402
    write_meerkat_multichoice_jsonl,
    write_supervised_csv,
)
from src.before.data_pipeline.data_saver import (  # noqa: E402
    MultiJsonlSingleFolderProcessor as BeforeMeerkat,
    SingleCsvProcessor as BeforeCsv,
)
from src.after.data_pipeline.data_saver import (  # noqa: E402
    MultiJsonlSingleFolderProcessor as AfterMeerkat,
    SingleCsvProcessor as AfterCsv,
)

SIZES = [1000, 5000, 20000, 50000, 100000]
REPEATS = 7
SEED = 42


def _hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def bench_supervised_csv(n: int, workdir: str) -> dict:
    raw = os.path.join(workdir, "raw")
    write_supervised_csv(os.path.join(raw, "Healthinfo.csv"), n, seed=SEED)
    out_before = os.path.join(workdir, "out_before")
    out_after = os.path.join(workdir, "out_after")

    before = BeforeCsv(raw_data_dir=raw, output_data_dir=out_before)
    after = AfterCsv(raw_data_dir=raw, output_data_dir=out_after)

    before.process_supervised_data("Healthinfo", "instruction", "output")
    after.process_supervised_data("Healthinfo", "instruction", "output")
    h_before = _hash_file(os.path.join(out_before, "Healthinfo.jsonl"))
    h_after = _hash_file(os.path.join(out_after, "Healthinfo.jsonl"))
    assert h_before == h_after, f"CSV output mismatch at n={n}: {h_before} != {h_after}"

    mb = measure(lambda: before.process_supervised_data("Healthinfo", "instruction", "output"), REPEATS)
    ma = measure(lambda: after.process_supervised_data("Healthinfo", "instruction", "output"), REPEATS)
    return {"scenario": "supervised_csv", "before": mb, "after": ma, "identical": True}


def bench_meerkat_jsonl(n: int, workdir: str) -> dict:
    raw = os.path.join(workdir, "raw")
    write_meerkat_multichoice_jsonl(os.path.join(raw, "Meerkat-Instructions", "MedQA-CoT.jsonl"), n, seed=SEED)
    out_before = os.path.join(workdir, "out_before")
    out_after = os.path.join(workdir, "out_after")

    before = BeforeMeerkat(raw_data_dir=raw, output_data_dir=out_before)
    after = AfterMeerkat(raw_data_dir=raw, output_data_dir=out_after)

    before.process_meerkat_instruction_jsonl()
    after.process_meerkat_instruction_jsonl()
    h_before = _hash_file(os.path.join(out_before, "Meerkat-Instructions", "MedQA-CoT.jsonl"))
    h_after = _hash_file(os.path.join(out_after, "Meerkat-Instructions", "MedQA-CoT.jsonl"))
    assert h_before == h_after, f"JSONL output mismatch at n={n}: {h_before} != {h_after}"

    mb = measure(lambda: before.process_meerkat_instruction_jsonl(), REPEATS)
    ma = measure(lambda: after.process_meerkat_instruction_jsonl(), REPEATS)
    return {"scenario": "meerkat_jsonl", "before": mb, "after": ma, "identical": True}


def main():
    rows = []
    for n in SIZES:
        for bench in (bench_supervised_csv, bench_meerkat_jsonl):
            workdir = tempfile.mkdtemp(prefix="pipe_bench_")
            try:
                r = bench(n, workdir)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            mb, ma = r["before"], r["after"]
            rows.append({
                "scenario": r["scenario"], "n": n,
                "before_mean_s": mb["mean_s"], "before_std_s": mb["std_s"], "before_peak_kib": mb["peak_kib"],
                "after_mean_s": ma["mean_s"], "after_std_s": ma["std_s"], "after_peak_kib": ma["peak_kib"],
                "speedup": mb["mean_s"] / ma["mean_s"] if ma["mean_s"] else 0.0,
                "mem_ratio": mb["peak_kib"] / ma["peak_kib"] if ma["peak_kib"] else 0.0,
                "identical": r["identical"],
            })
            print(f"[{r['scenario']:>14}] n={n:>7} "
                  f"time {mb['mean_s']*1000:8.2f}ms -> {ma['mean_s']*1000:8.2f}ms "
                  f"({rows[-1]['speedup']:.2f}x)  "
                  f"peak {mb['peak_kib']/1024:7.1f}MiB -> {ma['peak_kib']/1024:7.1f}MiB "
                  f"({rows[-1]['mem_ratio']:.2f}x)  identical={r['identical']}")

    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    csv_path = os.path.join(ROOT, "results", "pipeline_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    _plot(rows)


def _plot(rows):
    figdir = os.path.join(ROOT, "results", "figures")
    for scenario in ("supervised_csv", "meerkat_jsonl"):
        sub = [r for r in rows if r["scenario"] == scenario]
        ns = [r["n"] for r in sub]

        # time
        plt.figure(figsize=(6, 4))
        plt.errorbar(ns, [r["before_mean_s"] * 1000 for r in sub], yerr=[r["before_std_s"] * 1000 for r in sub],
                     marker="o", label="before")
        plt.errorbar(ns, [r["after_mean_s"] * 1000 for r in sub], yerr=[r["after_std_s"] * 1000 for r in sub],
                     marker="s", label="after")
        plt.xlabel("input rows (N)"); plt.ylabel("time (ms)")
        plt.title(f"pipeline time vs N — {scenario}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"pipeline_time_{scenario}.png"), dpi=300)
        plt.close()

        # memory
        plt.figure(figsize=(6, 4))
        plt.plot(ns, [r["before_peak_kib"] / 1024 for r in sub], marker="o", label="before")
        plt.plot(ns, [r["after_peak_kib"] / 1024 for r in sub], marker="s", label="after")
        plt.xlabel("input rows (N)"); plt.ylabel("peak memory (MiB)")
        plt.title(f"pipeline peak memory vs N — {scenario}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"pipeline_mem_{scenario}.png"), dpi=300)
        plt.close()
    print("wrote figures to", figdir)


if __name__ == "__main__":
    main()
