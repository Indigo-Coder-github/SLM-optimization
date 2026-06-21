"""AFTER: 최적화된 데이터 전처리.

핵심 변경 (before 대비)
- A 자료구조/복잡도
  * `match folder_name` 행별 재평가 → `registry.get_transformer()` 로 루프 밖 1회 디스패치(O(1)).
  * `_process_meerkat_multichoice` 의 `.find("(A)"~"(D)")` 를 **각 1회만** 계산.
- B 제너레이터/lazy
  * `pd.read_csv`+`iterrows()` → `csv.DictReader` 제너레이터(DataFrame 미적재).
  * `readlines()`/전체 list 적재 → 라인 스트리밍.
  * read→transform→record 를 **제너레이터 파이프라인**으로 구성하고 `save_jsonl` 가 스트리밍 기록
    ⇒ peak memory O(n) → O(1).
- C 클래스/SRP
  * `SupervisedData`/`UnsupervisedData` 에 `slots=True, frozen=True` 적용(메모리↓·불변).
  * 설정을 `ProcessorConfig`(frozen) 로 분리. 읽기/변환/기록 책임 분리.
- D 데코레이터
  * `@timed`/`@logged`(파이프라인 진입점), `save_jsonl` 의 `@validate_path`.

출력 JSONL 은 before 와 **바이트 동일**하도록 레코드 구조를 유지한다.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from src.common.chat_templates import (
    SYSTEM_PROMPT,
    korean_default_prompt,
    medqa_full_prompt,
)

from .decorators import logged, timed
from .registry import get_transformer
from .utils import iter_jsonl, save_jsonl


@dataclass(slots=True, frozen=True)
class SupervisedData:
    text: list[dict[str, str]]
    label: str | None = None
    subject: str | None = None


@dataclass(slots=True, frozen=True)
class UnsupervisedData:
    text: str


@dataclass(slots=True, frozen=True)
class ProcessorConfig:
    """설정값을 로직에서 분리(불변)."""
    raw_data_dir: str = "../data/raw_data"
    output_data_dir: str = "../data/processed_data"


# --- 레코드 변환(기록 책임) --------------------------------------------------
def to_record(item: SupervisedData | UnsupervisedData) -> dict:
    """Data 객체를 JSONL 레코드로 변환(before 의 save_jsonl 본문과 동일한 규칙)."""
    record: dict = {"messages": item.text}
    if isinstance(item, SupervisedData):
        if item.label:
            record["label"] = item.label
        if item.subject:
            record["subject"] = item.subject
    return record


# --- 읽기 책임(lazy 제너레이터) ---------------------------------------------
def iter_csv_rows(path: str) -> Iterator[dict]:
    """CSV 를 한 행씩 dict 로 내보낸다(전체 DataFrame 미적재)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def iter_txt_lines(path: str) -> Iterator[str]:
    """txt 를 한 줄씩 내보낸다(빈 줄 제외, before 와 동일 규칙)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                yield line.rstrip()


class DataProcessor:
    def __init__(self, raw_data_dir: str = "../data/raw_data", output_data_dir: str = "../data/processed_data"):
        self.config = ProcessorConfig(raw_data_dir, output_data_dir)

    @property
    def raw_data_dir(self) -> str:
        return self.config.raw_data_dir

    @property
    def output_data_dir(self) -> str:
        return self.config.output_data_dir

    @staticmethod
    def ensure_dir_exists(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_stream(self, items: Iterable[SupervisedData | UnsupervisedData], filepath: str) -> int:
        """Data 스트림을 곧장 디스크로 흘려보낸다(중간 리스트 없음)."""
        self.ensure_dir_exists(os.path.dirname(filepath))
        return save_jsonl((to_record(it) for it in items), filepath)


class SingleCsvProcessor(DataProcessor):
    @logged()
    @timed
    def process_supervised_data(self, csv_filename: str, input_column_name="instruction", output_column_name="output") -> None:
        path = os.path.join(self.raw_data_dir, f"{csv_filename}.csv")
        items = (
            SupervisedData(korean_default_prompt(row, input_column_name, output_column_name))
            for row in iter_csv_rows(path)
        )
        self.write_stream(items, os.path.join(self.output_data_dir, f"{csv_filename}.jsonl"))


class MultiCsvSingleFolderProcessor(DataProcessor):
    @logged()
    @timed
    def process_supervised_data(self, folder_name: str, label_name="label") -> None:
        folder_path = os.path.join(self.raw_data_dir, folder_name)
        transformer = get_transformer(folder_name)  # 루프 밖 1회 디스패치
        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue
            stem = filename.split(".")[0]
            csv_path = os.path.join(folder_path, filename)
            items = (
                SupervisedData(
                    text=transformer(row, stem),
                    label=row[label_name],
                    subject=row.get("subject"),
                )
                for row in iter_csv_rows(csv_path)
            )
            self.write_stream(items, os.path.join(self.output_data_dir, folder_name, filename.replace(".csv", ".jsonl")))

    @logged()
    @timed
    def process_unsupervised_data(self, folder_name: str) -> None:
        folder_path = os.path.join(self.raw_data_dir, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                items: Iterable[UnsupervisedData] = (
                    UnsupervisedData(text=row["Text"]) for row in iter_csv_rows(os.path.join(folder_path, filename))
                )
            elif filename.endswith("txt"):
                items = (UnsupervisedData(line) for line in iter_txt_lines(os.path.join(folder_path, filename)))
            else:
                raise ValueError(f"Unsupported file type or not available now: {filename}")
            self.write_stream(items, os.path.join(self.output_data_dir, folder_name, filename.split('.')[0] + ".jsonl"))


class MultiJsonlSingleFolderProcessor(DataProcessor):
    _MULTICHOICE_FILES = frozenset({"MedBooks-18-CoT.jsonl", "MedMCQA.jsonl", "MedQA-CoT.jsonl"})

    @logged()
    @timed
    def process_meerkat_instruction_jsonl(self) -> None:
        folder_path = os.path.join(self.raw_data_dir, "Meerkat-Instructions")
        for filename in os.listdir(folder_path):
            if not filename.endswith(".jsonl"):
                continue
            is_multichoice = filename in self._MULTICHOICE_FILES  # set 멤버십 O(1), 루프 밖 1회
            rows = iter_jsonl(os.path.join(folder_path, filename))  # lazy 스트리밍
            if is_multichoice:
                items = (SupervisedData(*self._process_meerkat_multichoice(r["messages"])) for r in rows)
            else:
                items = (SupervisedData(self._process_meerkat_instruction(r["messages"]), label="None") for r in rows)
            self.write_stream(items, os.path.join(self.output_data_dir, "Meerkat-Instructions", filename))

    @staticmethod
    def _process_meerkat_multichoice(row: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
        content = row[1]["content"]
        # (A)~(D) 위치를 각각 1회만 계산 (before 는 중복 호출)
        a = content.find("(A)")
        b = content.find("(B)")
        c = content.find("(C)")
        d = content.find("(D)")
        extracted_options = {
            "A": content[a + 3:b].strip(),
            "B": content[b + 3:c].strip(),
            "C": content[c + 3:d].strip(),
            "D": content[d + 3:].strip(),
        }
        extracted = {"question": content[:a].strip(), "options": extracted_options}

        answer_content = row[2]["content"]
        answer_phrase = "Therefore, the answer is ("
        another_answer_phrase = "The answer is ("
        option_idx = answer_content.find(answer_phrase) + len(answer_phrase)
        if option_idx < len(answer_phrase):
            option_idx = answer_content.find(another_answer_phrase) + len(another_answer_phrase)
        extracted_answer = answer_content[option_idx]

        chat_completion = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": medqa_full_prompt(extracted)["full_prompt"]},
            {"role": "assistant", "content": f"### Explanation:\n{answer_content}\n### Answer: {extracted_answer}"},
        ]
        return chat_completion, extracted_answer

    @staticmethod
    def _process_meerkat_instruction(row: list[dict[str, str]]) -> list[dict[str, str]]:
        chat_completion = [{"role": "system", "content": f"{SYSTEM_PROMPT}"}]
        chat_completion.append({"role": "user", "content": row[1]["content"] if row[1]["content"] else row[0]["content"]})
        chat_completion.append({"role": "assistant", "content": row[2]["content"]})
        if len(row) > 3:
            for i in range(3, len(row)):
                chat_completion.append({"role": "user", "content": row[i]["content"]})
        return chat_completion
