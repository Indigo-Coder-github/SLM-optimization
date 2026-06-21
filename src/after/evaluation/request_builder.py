"""AFTER: 요청 빌드/반복 책임 분리 (카테고리 B/C).

원본은 `repeated_requests.extend([req]*n)` 로 self-consistency 반복본을 만들고
`shuffle_options` 도 같은 메서드 안에 섞여 있었다. 여기서는 요청 생성/반복을 분리하고
반복을 `itertools` 로 표현한다. shuffle 은 before 와 동일 결과를 보장하기 위해
RNG 호출 순서를 그대로 유지한다.
"""

from __future__ import annotations

import re
from itertools import chain, repeat
from random import sample

_OPTION_LINE_RE = re.compile(r'^([A-E])\.\s(.*)$')


class RequestBuilder:
    def build(self, test_data_messages: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
        # 0-shot 요청. 원본과 동일하게 얕은 사본 리스트를 만든다.
        return list(test_data_messages)

    def repeat_for_self_consistency(self, requests: list[list[dict[str, str]]], n: int) -> list:
        """각 요청을 n회 반복(연속). itertools 로 중간 리스트 누적을 피한다."""
        if n == 1:
            return list(requests)
        return list(chain.from_iterable(repeat(req, n) for req in requests))

    def shuffle_options(self, requests: list[list[dict[str, str]]], num_questions: int, num_repeats: int):
        """보기 셔플(self-consistency). before 와 동일한 RNG 소비 순서를 유지."""
        flattend_requests: list[list[dict[str, str]]] = []
        mapping_tables: list[dict[str, str]] = []

        for q_id in range(num_questions):
            for t_id in range(num_repeats):
                request_id = q_id * num_repeats + t_id
                trial = requests[request_id]
                lines = trial[-1]["content"].split("\n")

                # 역순 복사본(lines[::-1]) 대신 인덱스 역방향 스캔
                indicator_idx = -1
                for idx in range(len(lines) - 1, -1, -1):
                    line = lines[idx]
                    if "### 보기" in line or "### Options" in line:
                        indicator_idx = idx
                        break
                if indicator_idx < 0:
                    raise ValueError("Invalid Format: Not Found '### Options'")

                option_lines = []
                option_idx = indicator_idx + 1
                while option_idx < len(lines):
                    m = _OPTION_LINE_RE.match(lines[option_idx].strip())
                    if not m:
                        break
                    option_lines.append((m.group(1), m.group(2).strip()))
                    option_idx += 1

                original_options_map = {idx: txt for idx, txt in option_lines}
                shuffled = sample(list(original_options_map.keys()), len(original_options_map))
                mapping_table = {origin: new for origin, new in zip(original_options_map.keys(), shuffled)}

                reconstructed_options = [f"{new}. {original_options_map[origin]}" for origin, new in mapping_table.items()]
                reconstructed_options = sorted(reconstructed_options, key=lambda x: x[0])

                new_lines = lines[:indicator_idx + 1] + reconstructed_options
                new_chat_completion = trial[:-1] + [{"role": "user", "content": "\n".join(new_lines)}]

                flattend_requests.append(new_chat_completion)
                mapping_tables.append({new: origin for origin, new in mapping_table.items()})

        return flattend_requests, mapping_tables
