"""BEFORE: 원본 `src/evaluation/evaluator.py::Evaluator.evaluate_model` 의 baseline.

GPU/vLLM 의존을 없애기 위해 모델 호출만 주입된 `backend` 로 추상화했다.
그 외 구조(한 메서드에 모든 책임 혼재, `pop(0)` budget forcing, `mode()` voting)는
원본을 그대로 유지하여 최적화 전 상태를 충실히 재현한다.

backend 는 vLLM `LLM` 과 동일한 형태로 다음을 제공한다고 가정한다(duck typing):
    backend.chat(requests) -> list[Out]
    backend.generate(prompts) -> list[Out]
    backend.get_tokenizer().apply_chat_template(messages, tokenize=False) -> str
where Out.outputs[0].text: str, Out.outputs[0].token_ids: list[int]
"""

from statistics import StatisticsError, mode

from src.before.data_pipeline.utils import extract_answer_from_response
from src.before.evaluation.prompt_engineer import PromptEngineer


class Evaluator:
    def __init__(self, backend, num_repeats: int = 1):
        # 설정과 런타임 의존(backend)이 생성자에 함께 묶여 있다.
        self.backend = backend
        self.num_repeats = num_repeats

    def evaluate_model(self, test_data_messages: list[list[dict[str, str]]], labels: list[str],
                       shuffle_options: bool = False, test_time_scaling: bool = False,
                       token_budget: int = 4096, ignore_eos_num: int = 1):
        """요청 빌드·반복·생성·budget forcing·추출·voting·정확도를 한 메서드에서 수행."""
        prompt_engineer = PromptEngineer(test_data_messages)
        vllm_requests = prompt_engineer.build_requests()
        num_questions = len(vllm_requests)

        # self-consistency 반복: 중간 리스트 대량 생성(extend([req]*n))
        if shuffle_options:
            assert self.num_repeats > 1, "num_repeats must be > 1 when shuffle_options is True"
            repeated_requests = list()
            for req in vllm_requests:
                repeated_requests.extend([req] * self.num_repeats)
            vllm_requests, mapping_tables = prompt_engineer.shuffle_options(repeated_requests, num_questions, self.num_repeats)
        elif self.num_repeats > 1:
            repeated_requests = list()
            for req in vllm_requests:
                repeated_requests.extend([req] * self.num_repeats)
            vllm_requests = repeated_requests

        outputs = self.backend.chat(vllm_requests)

        if test_time_scaling:
            outputs = self._budget_forcing(outputs, vllm_requests, ignore_eos_num, token_budget)
        else:
            outputs = [out.outputs[0].text for out in outputs]

        # 답변 추출 후 majority voting(self-consistency)
        resp_text_list, filtered_resps_list, final_resps = list(), list(), list()
        num_questions = len(outputs) // self.num_repeats
        for q_id in range(num_questions):
            resp_texts, filtered_resps = list(), list()
            for t_id in range(self.num_repeats):
                output = outputs[q_id * self.num_repeats + t_id]
                resp_texts.append(output)
                filtered_resp = extract_answer_from_response(output)
                if shuffle_options:
                    mapping = mapping_tables[q_id * self.num_repeats + t_id]
                    reverted_answer = mapping[filtered_resp] if filtered_resp in mapping else "invalid"
                    filtered_resps.append(reverted_answer)
                else:
                    filtered_resps.append(filtered_resp if filtered_resp in ["A", "B", "C", "D", "E"] else "invalid")
            resp_text_list.append(resp_texts)
            filtered_resps_list.append(filtered_resps)
            try:
                final_resps.append(mode([i for i in filtered_resps if i != "invalid"]))
            except StatisticsError:
                final_resps.append("invalid")

        # 정확도(간단 구현; 원본은 sklearn.accuracy_score)
        correct = sum(1 for p, l in zip(final_resps, labels) if p == l)
        acc = round(correct / len(labels), 5) if labels else 0.0
        return acc, final_resps

    def _budget_forcing(self, initial_outputs, chat_requests, ignore_eos_num: int = 1, token_budget: int = 4096):
        if ignore_eos_num < 1:
            raise ValueError("ignore_eos_num must be larger than 0")
        tokenizer = self.backend.get_tokenizer()

        ignore_str = "<think>Wait"
        remained_budgets = [token_budget - len(o.outputs[0].token_ids) for o in initial_outputs]

        generation_by_model = [{"role": "assistant", "content": out.outputs[0].text} for out in initial_outputs]
        concatenated_chat_completion = [i + [j] for i, j in zip(chat_requests, generation_by_model)]

        for num in range(ignore_eos_num):
            budget_remained = [rb > 0 for rb in remained_budgets]
            if num == 0:
                concatenated_chat_completion = [tokenizer.apply_chat_template(c, tokenize=False) for c in concatenated_chat_completion]
            for idx, (chat_completion, is_remained) in enumerate(zip(concatenated_chat_completion, budget_remained)):
                if is_remained:
                    concatenated_chat_completion[idx] = chat_completion + ignore_str

            longer_outputs = self.backend.generate(concatenated_chat_completion)
            longer_outputs_text = [out.outputs[0].text for out in longer_outputs]
            longer_outputs_token_ids = [out.outputs[0].token_ids for out in longer_outputs]

            for idx, is_remained in enumerate(budget_remained):
                if is_remained:
                    concatenated_chat_completion[idx] += longer_outputs_text[idx]
                    # list.pop(0) 를 루프에서 호출 → 매 호출 O(n), 전체 O(n^2)
                    remained_budgets[idx] -= len(longer_outputs_token_ids.pop(0))

        concatenated_chat_completion = [i + "### Answer:" for i in concatenated_chat_completion]
        last_generation = self.backend.generate(concatenated_chat_completion)
        return [out.outputs[0].text for out in last_generation]
