"""AFTER: 답변 추출 + self-consistency 집계 책임 분리 (카테고리 A/B).

- majority voting 을 `statistics.mode`(+예외 처리) 대신 `collections.Counter` 로 수행 → O(n),
  빈 표는 결정적으로 "invalid" 처리(정보 손실 없음).
- 그룹 단위 순회를 `outputs[q*r + t]` 수동 인덱싱 대신 `itertools.batched` 로 표현.
- before 와 **동일한 final answer 시퀀스**를 산출한다.
"""

from __future__ import annotations

from collections import Counter
from itertools import batched

from src.after.data_pipeline.utils import extract_answer_from_response, is_valid_letter


class AnswerAggregator:
    def __init__(self, num_repeats: int):
        self.num_repeats = num_repeats

    @staticmethod
    def _majority(filtered: list[str]) -> str:
        valid = [x for x in filtered if x != "invalid"]
        if not valid:
            return "invalid"
        # Counter 는 삽입 순서를 보존하므로 동률 시 first-seen 이 선택됨(mode 와 동일).
        return Counter(valid).most_common(1)[0][0]

    def aggregate(self, outputs: list[str], shuffle_options: bool = False, mapping_tables=None):
        resp_text_list, filtered_resps_list, final_resps = [], [], []

        for group_idx, group in enumerate(batched(outputs, self.num_repeats)):
            resp_texts, filtered_resps = [], []
            for t_id, output in enumerate(group):
                resp_texts.append(output)
                filtered = extract_answer_from_response(output)
                if shuffle_options:
                    mapping = mapping_tables[group_idx * self.num_repeats + t_id]
                    filtered_resps.append(mapping.get(filtered, "invalid"))
                else:
                    filtered_resps.append(filtered if is_valid_letter(filtered) else "invalid")
            resp_text_list.append(resp_texts)
            filtered_resps_list.append(filtered_resps)
            final_resps.append(self._majority(filtered_resps))

        return resp_text_list, filtered_resps_list, final_resps
