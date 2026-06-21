"""AFTER: 재설계된 Evaluator (카테고리 A/B/C/D 종합).

원본의 God-method `evaluate_model` 을 협력 객체로 분해한다:
  RequestBuilder  — 요청 빌드/반복/셔플
  GenerationBackend(Protocol) — 모델 호출(vLLM ↔ Fake 교체 가능)
  BudgetForcer    — test-time scaling(budget forcing)
  AnswerAggregator — 추출 + self-consistency voting
  ResultSink(Protocol) — 결과 로깅(wandb 분리)
설정은 `EvalConfig`(frozen) 로 분리. 결과(정확도·final answers)는 before 와 동일하다.
"""

from __future__ import annotations

from collections import deque

from .aggregator import AnswerAggregator
from .backends import GenerationBackend, NullSink, ResultSink
from .config import EvalConfig
from .decorators import retry, timed
from .request_builder import RequestBuilder


class BudgetForcer:
    """test-time scaling 의 budget forcing 책임만 담당."""

    def __init__(self, backend: GenerationBackend, config: EvalConfig):
        self.backend = backend
        self.config = config

    @timed
    def apply(self, initial_outputs, chat_requests) -> list[str]:
        ignore_eos_num = self.config.ignore_eos_num
        token_budget = self.config.token_budget
        if ignore_eos_num < 1:
            raise ValueError("ignore_eos_num must be larger than 0")
        tokenizer = self.backend.get_tokenizer()

        ignore_str = "<think>Wait"
        remained_budgets = [token_budget - len(o.outputs[0].token_ids) for o in initial_outputs]
        generation_by_model = [{"role": "assistant", "content": out.outputs[0].text} for out in initial_outputs]
        concatenated = [req + [gen] for req, gen in zip(chat_requests, generation_by_model)]

        for num in range(ignore_eos_num):
            budget_remained = [rb > 0 for rb in remained_budgets]
            if num == 0:
                concatenated = [tokenizer.apply_chat_template(c, tokenize=False) for c in concatenated]
            for idx, remained in enumerate(budget_remained):
                if remained:
                    concatenated[idx] = concatenated[idx] + ignore_str

            longer_outputs = self._generate(concatenated)
            longer_texts = [out.outputs[0].text for out in longer_outputs]
            # pop(0) (O(n^2)) 대신 deque.popleft (O(1)) → 전체 O(n)
            token_id_queue = deque(out.outputs[0].token_ids for out in longer_outputs)

            for idx, remained in enumerate(budget_remained):
                if remained:
                    concatenated[idx] += longer_texts[idx]
                    remained_budgets[idx] -= len(token_id_queue.popleft())

        concatenated = [c + "### Answer:" for c in concatenated]
        last_generation = self._generate(concatenated)
        return [out.outputs[0].text for out in last_generation]

    @retry(max_attempts=3, base_delay=0.0)
    def _generate(self, prompts):
        return self.backend.generate(prompts)


class Evaluator:
    def __init__(self, backend: GenerationBackend, config: EvalConfig, sink: ResultSink | None = None):
        self.backend = backend
        self.config = config
        self.sink = sink or NullSink()
        self.builder = RequestBuilder()
        self.aggregator = AnswerAggregator(config.num_repeats)
        self.budget_forcer = BudgetForcer(backend, config)

    @retry(max_attempts=3, base_delay=0.0)
    def _chat(self, requests):
        return self.backend.chat(requests)

    @timed
    def evaluate_model(self, test_data_messages, labels, data_name: str = "eval"):
        cfg = self.config
        requests = self.builder.build(test_data_messages)
        num_questions = len(requests)

        mapping_tables = None
        if cfg.shuffle_options:
            repeated = self.builder.repeat_for_self_consistency(requests, cfg.num_repeats)
            requests, mapping_tables = self.builder.shuffle_options(repeated, num_questions, cfg.num_repeats)
        elif cfg.num_repeats > 1:
            requests = self.builder.repeat_for_self_consistency(requests, cfg.num_repeats)

        raw_outputs = self._chat(requests)

        if cfg.test_time_scaling:
            outputs = self.budget_forcer.apply(raw_outputs, requests)
        else:
            outputs = [out.outputs[0].text for out in raw_outputs]

        resp_text_list, filtered_resps_list, final_resps = self.aggregator.aggregate(
            outputs, shuffle_options=cfg.shuffle_options, mapping_tables=mapping_tables)

        self.sink.log_results(data_name, resp_text_list, filtered_resps_list, final_resps)

        correct = sum(1 for p, l in zip(final_resps, labels) if p == l)
        acc = round(correct / len(labels), 5) if labels else 0.0
        return acc, final_resps
