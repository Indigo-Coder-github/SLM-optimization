"""합성(synthetic) vs 실제(real) 데이터 전처리 벤치마크.

목적: '성능은 데이터 내용이 아니라 크기·스키마에 좌우된다'와 '최적화는 실제
데이터에서도 출력이 동일하다'를 직접 검증한다. HuggingFace 의 실제 의료 벤치마크
(MedQA-USMLE 4지선다)를 받아 파이프라인이 기대하는 동일 스키마(CSV instruction/output,
Meerkat JSONL messages)로 변환한 뒤, synthetic 과 같은 N 에서 before/after 를 측정해
(1) speedup·memory 비율이 비슷한지, (2) before/after 출력이 md5 까지 같은지 비교한다.

결과: results/synth_vs_real.csv, results/figures/synth_vs_real.png
인터넷/datasets 가 없으면 안전하게 건너뛴다(보고서는 그대로 유효).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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

REPEATS = 7
SEED = 42
DATASET_CANDIDATES = ["GBaker/MedQA-USMLE-4-options", "augtoma/medqa_usmle"]
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", str(s)).strip()


def _hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- 실제 데이터 적재(MedQA 4지선다) ----------
def load_real():
    from datasets import load_dataset
    last = None
    for name in DATASET_CANDIDATES:
        try:
            print(f"[real] load_dataset({name}) ...", flush=True)
            ds = load_dataset(name, split="train")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[real] {name} 실패: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        items = []
        for ex in ds:
            q = ex.get("question")
            opts = ex.get("options")
            ans = ex.get("answer_idx")
            if isinstance(opts, dict):
                a, b, c, d = opts.get("A"), opts.get("B"), opts.get("C"), opts.get("D")
            else:
                a = b = c = d = None
            if not (q and a and b and c and d and ans in ("A", "B", "C", "D")):
                continue
            items.append({"q": _clean(q), "A": _clean(a), "B": _clean(b),
                          "C": _clean(c), "D": _clean(d), "ans": ans})
        if items:
            print(f"[real] {name}: {len(items)} usable examples", flush=True)
            return items
    raise RuntimeError(f"실제 데이터 적재 실패: {last}")


# ---------- 실제 데이터를 파이프라인 스키마로 기록 ----------
def write_real_csv(items):
    def _w(path, n, seed=SEED):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instruction", "output"])
            for it in items[:n]:
                instr = f"{it['q']} (A) {it['A']} (B) {it['B']} (C) {it['C']} (D) {it['D']}"
                w.writerow([instr, it["ans"]])
        return path
    return _w


def write_real_meerkat(items):
    def _w(path, n, seed=SEED):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for it in items[:n]:
                user = f"{it['q']} (A) {it['A']} (B) {it['B']} (C) {it['C']} (D) {it['D']}"
                assistant = f"Step by step reasoning. Therefore, the answer is ({it['ans']})"
                row = {"messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
    return _w


# ---------- 한 시나리오 측정(before/after) ----------
def run_csv(write_fn, source, n):
    workdir = tempfile.mkdtemp(prefix="svr_csv_")
    try:
        raw = os.path.join(workdir, "raw")
        write_fn(os.path.join(raw, "Healthinfo.csv"), n)
        ob, oa = os.path.join(workdir, "ob"), os.path.join(workdir, "oa")
        before, after = BeforeCsv(raw_data_dir=raw, output_data_dir=ob), AfterCsv(raw_data_dir=raw, output_data_dir=oa)
        before.process_supervised_data("Healthinfo", "instruction", "output")
        after.process_supervised_data("Healthinfo", "instruction", "output")
        identical = _hash_file(os.path.join(ob, "Healthinfo.jsonl")) == _hash_file(os.path.join(oa, "Healthinfo.jsonl"))
        mb = measure(lambda: before.process_supervised_data("Healthinfo", "instruction", "output"), REPEATS)
        ma = measure(lambda: after.process_supervised_data("Healthinfo", "instruction", "output"), REPEATS)
        return _row("supervised_csv", source, n, mb, ma, identical)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_jsonl(write_fn, source, n):
    workdir = tempfile.mkdtemp(prefix="svr_jsonl_")
    try:
        raw = os.path.join(workdir, "raw")
        write_fn(os.path.join(raw, "Meerkat-Instructions", "MedQA-CoT.jsonl"), n)
        ob, oa = os.path.join(workdir, "ob"), os.path.join(workdir, "oa")
        before, after = BeforeMeerkat(raw_data_dir=raw, output_data_dir=ob), AfterMeerkat(raw_data_dir=raw, output_data_dir=oa)
        before.process_meerkat_instruction_jsonl()
        after.process_meerkat_instruction_jsonl()
        p = os.path.join("Meerkat-Instructions", "MedQA-CoT.jsonl")
        identical = _hash_file(os.path.join(ob, p)) == _hash_file(os.path.join(oa, p))
        mb = measure(lambda: before.process_meerkat_instruction_jsonl(), REPEATS)
        ma = measure(lambda: after.process_meerkat_instruction_jsonl(), REPEATS)
        return _row("meerkat_jsonl", source, n, mb, ma, identical)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _row(scenario, source, n, mb, ma, identical):
    return {"scenario": scenario, "source": source, "n": n,
            "before_mean_s": mb["mean_s"], "after_mean_s": ma["mean_s"],
            "speedup": mb["mean_s"] / ma["mean_s"] if ma["mean_s"] else 0.0,
            "before_peak_kib": mb["peak_kib"], "after_peak_kib": ma["peak_kib"],
            "mem_ratio": mb["peak_kib"] / ma["peak_kib"] if ma["peak_kib"] else 0.0,
            "identical": identical}


def main():
    real = load_real()
    maxn = len(real)
    sizes = [s for s in (1000, 5000, 10000) if s <= maxn]
    if not sizes or sizes[-1] < maxn < 12000:
        sizes = sorted(set(sizes + [maxn]))
    print(f"[svr] sizes={sizes} (real max {maxn})", flush=True)

    real_csv_w, real_jsonl_w = write_real_csv(real), write_real_meerkat(real)
    syn_csv_w = lambda p, n: write_supervised_csv(p, n, seed=SEED)
    syn_jsonl_w = lambda p, n: write_meerkat_multichoice_jsonl(p, n, seed=SEED)

    rows = []
    for n in sizes:
        for source, cw, jw in (("synthetic", syn_csv_w, syn_jsonl_w), ("real", real_csv_w, real_jsonl_w)):
            for fn, scen in ((run_csv, "supervised_csv"), (run_jsonl, "meerkat_jsonl")):
                r = fn(cw if scen == "supervised_csv" else jw, source, n)
                rows.append(r)
                print(f"[{scen:>14} | {source:>9}] n={n:>6}  "
                      f"{r['before_mean_s']*1000:8.2f}->{r['after_mean_s']*1000:8.2f}ms "
                      f"({r['speedup']:.2f}x)  mem {r['before_peak_kib']/1024:6.1f}->{r['after_peak_kib']/1024:6.1f}MiB "
                      f"({r['mem_ratio']:.2f}x)  identical={r['identical']}", flush=True)

    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    out = os.path.join(ROOT, "results", "synth_vs_real.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)
    _plot(rows)


def _plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))

    def _panel(ax, scenario, key, ylabel, title):
        for source, style in (("synthetic", "o--"), ("real", "s-")):
            sub = sorted([r for r in rows if r["scenario"] == scenario and r["source"] == source], key=lambda r: r["n"])
            if sub:
                ax.plot([r["n"] for r in sub], [r[key] for r in sub], style, label=source)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("input rows (N)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _panel(axes[0], "supervised_csv", "speedup", "speedup (x)", "(a) CSV: synthetic vs real speedup")
    _panel(axes[1], "meerkat_jsonl", "mem_ratio", "memory ratio (x)", "(b) JSONL: synthetic vs real memory gain")
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "synth_vs_real.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
