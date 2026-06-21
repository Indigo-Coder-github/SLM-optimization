"""BEFORE: 원본 `src/evaluation/prompt_engineer.py` 의 CPU-bound 부분 baseline.

few-shot(kNN)은 chromadb·임베딩 의존이라 제외하고, GPU 없이 측정 가능한
0-shot 요청 빌드와 `shuffle_options`(self-consistency 옵션 셔플)만 충실히 옮긴다.
"""

import re
from random import sample


class PromptEngineer:
    def __init__(self, test_data_messages: list[list[dict[str, str]]]):
        # 원본은 datasets.Dataset 을 받지만, 측정용으로 messages 리스트만 받는다.
        self.test_data_messages = test_data_messages

    def build_requests(self) -> list[list[dict[str, str]]]:
        # 0-shot: 원본과 동일하게 list comprehension 으로 새 리스트 생성
        return [i for i in self.test_data_messages]

    def shuffle_options(self, requests: list[list[dict[str, str]]], num_questions: int, num_repeats: int):
        flattend_requests: list[list[dict[str, str]]] = list()
        mapping_tables: list[dict[str, str]] = list()

        option_line_re = re.compile(r'^([A-E])\.\s(.*)$')

        for q_id in range(num_questions):
            for t_id in range(num_repeats):
                request_id = q_id * num_repeats + t_id
                trial = requests[request_id]
                lines = trial[-1]["content"].split("\n")

                # 마지막 보기 지시자 탐색 — 역순 복사본 생성(lines[::-1])
                for idx, line in enumerate(lines[::-1]):
                    if "### 보기" in line or "### Options" in line:
                        indicator_idx = len(lines) - 1 - idx
                        break
                else:
                    raise ValueError("Invalid Format: Not Found '### Options'")

                option_lines = list()
                option_idx = indicator_idx + 1
                while option_idx < len(lines):
                    m = option_line_re.match(lines[option_idx].strip())
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
                new_line = "\n".join(new_lines)
                new_chat_completion = trial[:-1] + [{"role": "user", "content": new_line}]

                flattend_requests.append(new_chat_completion)
                mapping_tables.append({new: origin for origin, new in mapping_table.items()})

        return flattend_requests, mapping_tables
