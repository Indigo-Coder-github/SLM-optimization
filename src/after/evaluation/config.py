"""AFTER: 평가 설정 객체 (카테고리 C — config/state 분리).

원본 `Evaluator.__init__` 은 vLLM 초기화·SamplingParams·정책 플래그를 한데 묶었다.
여기서는 **불변 설정**만 `EvalConfig` 로 분리하고, 런타임 의존(backend)은 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EvalConfig:
    num_repeats: int = 1
    shuffle_options: bool = False
    test_time_scaling: bool = False
    token_budget: int = 4096
    ignore_eos_num: int = 1

    def __post_init__(self):
        if self.num_repeats < 1:
            raise ValueError("num_repeats must be >= 1")
        if self.shuffle_options and self.num_repeats <= 1:
            raise ValueError("num_repeats must be > 1 when shuffle_options is True")
