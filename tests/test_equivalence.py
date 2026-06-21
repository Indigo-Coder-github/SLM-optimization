"""정확성·재현성 검증 테스트 (pytest 불필요 — `python tests/test_equivalence.py` 로 실행).

과제 목적의 "재현성"과 유의사항의 "최적화 전후 비교 가능"을 자동으로 보장한다.
검증 항목
  1) 전처리: before/after 가 동일 입력에 대해 바이트 동일한 JSONL 을 생성.
  2) Evaluator: before/after 가 동일한 final answer 시퀀스를 산출(self-consistency, budget forcing).
  3) 데코레이터: functools.wraps 가 __name__/__doc__/signature 를 보존(메타데이터 비손상).
  4) validate_path: 잘못된 확장자에 대해 예외를 숨기지 않고 올바르게 raise.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmark.make_synthetic_data import (  # noqa: E402
    write_meerkat_multichoice_jsonl, write_supervised_csv)
from benchmark.make_synthetic_responses import FakeBackend, build_eval_messages  # noqa: E402


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def test_pipeline_outputs_identical():
    from src.before.data_pipeline.data_saver import (
        MultiJsonlSingleFolderProcessor as BM, SingleCsvProcessor as BC)
    from src.after.data_pipeline.data_saver import (
        MultiJsonlSingleFolderProcessor as AM, SingleCsvProcessor as AC)

    wd = tempfile.mkdtemp(prefix="eq_")
    raw = os.path.join(wd, "raw")
    write_supervised_csv(os.path.join(raw, "H.csv"), 300, seed=7)
    write_meerkat_multichoice_jsonl(os.path.join(raw, "Meerkat-Instructions", "MedQA-CoT.jsonl"), 300, seed=7)

    BC(raw_data_dir=raw, output_data_dir=os.path.join(wd, "b")).process_supervised_data("H", "instruction", "output")
    AC(raw_data_dir=raw, output_data_dir=os.path.join(wd, "a")).process_supervised_data("H", "instruction", "output")
    assert _md5(os.path.join(wd, "b", "H.jsonl")) == _md5(os.path.join(wd, "a", "H.jsonl")), "CSV 출력 불일치"

    BM(raw_data_dir=raw, output_data_dir=os.path.join(wd, "b")).process_meerkat_instruction_jsonl()
    AM(raw_data_dir=raw, output_data_dir=os.path.join(wd, "a")).process_meerkat_instruction_jsonl()
    bp = os.path.join(wd, "b", "Meerkat-Instructions", "MedQA-CoT.jsonl")
    ap = os.path.join(wd, "a", "Meerkat-Instructions", "MedQA-CoT.jsonl")
    assert _md5(bp) == _md5(ap), "JSONL 출력 불일치"


def test_evaluator_outputs_identical():
    from src.before.evaluation.evaluator_core import Evaluator as BE
    from src.after.evaluation.config import EvalConfig
    from src.after.evaluation.evaluator import Evaluator as AE

    msgs, labels = build_eval_messages(150, seed=99)

    for r in (1, 5):
        ab, fb = BE(FakeBackend(), num_repeats=r).evaluate_model(msgs, labels)
        aa, fa = AE(FakeBackend(), EvalConfig(num_repeats=r)).evaluate_model(msgs, labels)
        assert fb == fa and ab == aa, f"self-consistency 불일치 (r={r})"

    ab, fb = BE(FakeBackend(), num_repeats=1).evaluate_model(
        msgs, labels, test_time_scaling=True, token_budget=100000, ignore_eos_num=3)
    aa, fa = AE(FakeBackend(), EvalConfig(num_repeats=1, test_time_scaling=True,
                                          token_budget=100000, ignore_eos_num=3)).evaluate_model(msgs, labels)
    assert fb == fa and ab == aa, "budget forcing 불일치"


def test_decorator_preserves_metadata():
    from src.after.data_pipeline.decorators import logged, timed, validate_path

    @timed
    def foo(a, b):
        """docstring-foo"""
        return a + b

    assert foo.__name__ == "foo", "timed 가 __name__ 손상"
    assert foo.__doc__ == "docstring-foo", "timed 가 __doc__ 손상"
    assert list(inspect.signature(foo).parameters) == ["a", "b"], "timed 가 signature 손상"

    @logged()
    def bar(x):
        """docstring-bar"""
        return x

    assert bar.__name__ == "bar" and bar.__doc__ == "docstring-bar", "logged 가 metadata 손상"

    @validate_path(suffixes=(".jsonl",), arg_index=0)
    def baz(path):
        """docstring-baz"""
        return path

    assert baz.__name__ == "baz" and baz.__doc__ == "docstring-baz", "validate_path 가 metadata 손상"


def test_validate_path_raises_not_hides():
    from src.after.data_pipeline.decorators import validate_path

    @validate_path(suffixes=(".jsonl",), arg_index=0)
    def writer(path):
        return path

    writer("ok.jsonl")  # 정상
    raised = False
    try:
        writer("bad.txt")
    except ValueError:
        raised = True
    assert raised, "잘못된 확장자에 대해 예외가 발생하지 않음(예외 은닉)"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
