"""BEFORE: 원본 SLM `src/data_processing/data_saver.py` 의 baseline.

원본과의 유일한 차이는 입출력 디렉터리를 생성자 인자로 받도록 한 점이다
(원본은 `self.raw_data_dir = "../data/raw_data"` 처럼 하드코딩).
측정되는 알고리즘 자체는 원본과 동일하다.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.chat_templates import (
    SYSTEM_PROMPT,
    korean_default_prompt,
    kormedmcqa_and_aihub_chat_template,
    medqa_full_prompt,
    pubmedqa_chat_template,
)

from .utils import save_jsonl


@dataclass
class SupervisedData:
    """처리된 지도학습 데이터 구조 (slots/frozen 미적용 — 원본 그대로)."""
    text: list[dict[str, str]]
    label: str | None = None
    subject: str | None = None


@dataclass
class UnsupervisedData:
    """비지도 학습 데이터 구조."""
    text: str


class DataProcessor:
    def __init__(self, raw_data_dir: str = "../data/raw_data", output_data_dir: str = "../data/processed_data"):
        self.raw_data_dir = raw_data_dir
        self.output_data_dir = output_data_dir

    def ensure_dir_exists(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_jsonl(self, data: list, filepath: str):
        """리스트 전체를 `output_items` 에 누적한 뒤 한 번에 저장."""
        self.ensure_dir_exists(os.path.dirname(filepath))
        output_items = list()
        for item in data:
            output_item: dict = {"messages": item.text}
            if isinstance(item, SupervisedData) and item.label:
                output_item["label"] = item.label
            if isinstance(item, SupervisedData) and item.subject:
                output_item["subject"] = item.subject
            output_items.append(output_item)
        save_jsonl(output_items, filepath)


class SingleCsvProcessor(DataProcessor):
    """단일 CSV 처리."""

    def process_supervised_data(self, csv_filename: str, input_column_name="instruction", output_column_name="output") -> None:
        df = pd.read_csv(os.path.join(self.raw_data_dir, f"{csv_filename}.csv"))
        processed_data = []
        for _, row in df.iterrows():
            chat_messages = korean_default_prompt(row.to_dict(), input_column_name, output_column_name)
            processed_data.append(SupervisedData(chat_messages, None))
        self.save_jsonl(processed_data, os.path.join(self.output_data_dir, f"{csv_filename}.jsonl"))


class MultiCsvSingleFolderProcessor(DataProcessor):
    """다중 CSV, 단일 폴더 처리."""

    def process_supervised_data(self, folder_name: str, label_name="label") -> None:
        folder_path = os.path.join(self.raw_data_dir, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(folder_path, filename)
                df = pd.read_csv(csv_path)
                processed_data = []
                for _, row in df.iterrows():
                    match folder_name:  # 행마다 match 디스패치 재평가
                        case "KorMedMCQA_Gemini":
                            chat_messages = kormedmcqa_and_aihub_chat_template(row.to_dict(), filename.split(".")[0])
                            processed_data.append(SupervisedData(text=chat_messages, label=row[label_name], subject=row["subject"]))
                        case "PubMedQA":
                            chat_messages = pubmedqa_chat_template(row.to_dict())
                            processed_data.append(SupervisedData(text=chat_messages, label=row[label_name]))
                        case "AI_Hub_Multi_Choice":
                            chat_messages = kormedmcqa_and_aihub_chat_template(row.to_dict(), filename.split(".")[0])
                            processed_data.append(SupervisedData(text=chat_messages, label=row[label_name]))
                        case _:
                            raise ValueError(f"Unknown folder name: {folder_name}")
                self.save_jsonl(processed_data, os.path.join(self.output_data_dir, folder_name, filename.replace(".csv", ".jsonl")))

    def process_unsupervised_data(self, folder_name: str) -> None:
        folder_path = os.path.join(self.raw_data_dir, folder_name)
        for filename in os.listdir(folder_path):
            processed_data = []
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                for _, row in df.iterrows():
                    processed_data.append(UnsupervisedData(text=row["Text"]))
            elif filename.endswith("txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    processed_data = [UnsupervisedData(line.rstrip()) for line in f.readlines() if line != "\n"]
            else:
                raise ValueError(f"Unsupported file type or not available now: {filename}")
            self.save_jsonl(processed_data, os.path.join(self.output_data_dir, folder_name, filename.split('.')[0] + ".jsonl"))


class MultiJsonlSingleFolderProcessor(DataProcessor):
    """다중 JSONL, 단일 폴더 처리 (Meerkat)."""

    def process_meerkat_instruction_jsonl(self):
        folder_path = os.path.join(self.raw_data_dir, "Meerkat-Instructions")
        for filename in os.listdir(folder_path):
            if filename.endswith(".jsonl"):
                processed_data = []
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    # 파일 전체를 readlines 로 적재 후 리스트 comprehension
                    file_contents = [json.loads(line.rstrip()) for line in f.readlines()]
                for row in file_contents:
                    match filename:
                        case "MedBooks-18-CoT.jsonl" | "MedMCQA.jsonl" | "MedQA-CoT.jsonl":
                            chat_messages, extracted_answer = self._process_meerkat_multichoice(row["messages"])
                            processed_data.append(SupervisedData(text=chat_messages, label=extracted_answer))
                        case _:
                            chat_messages = self._process_meerkat_instruction(row["messages"])
                            processed_data.append(SupervisedData(text=chat_messages, label="None"))
                self.save_jsonl(processed_data, os.path.join(self.output_data_dir, "Meerkat-Instructions", filename))

    def _process_meerkat_multichoice(self, row: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
        chat_completion = list()
        chat_completion.append({"role": "system", "content": f"{SYSTEM_PROMPT}"})

        # 동일한 .find("(A)") 등을 여러 번 중복 호출
        extracted_question = row[1]["content"][:row[1]["content"].find("(A)")].strip()
        extracted_options = {
            "A": row[1]["content"][row[1]["content"].find("(A)") + len("(A)"):row[1]["content"].find("(B)")].strip(),
            "B": row[1]["content"][row[1]["content"].find("(B)") + len("(B)"):row[1]["content"].find("(C)")].strip(),
            "C": row[1]["content"][row[1]["content"].find("(C)") + len("(C)"):row[1]["content"].find("(D)")].strip(),
            "D": row[1]["content"][row[1]["content"].find("(D)") + len("(D)"):].strip(),
        }
        extracted_question_and_options = {"question": extracted_question, "options": extracted_options}

        answer_phrase = "Therefore, the answer is ("
        another_answer_phrase = "The answer is ("
        option_idx = row[2]["content"].find(answer_phrase) + len(answer_phrase)
        if option_idx < len(answer_phrase):
            option_idx = row[2]["content"].find(another_answer_phrase) + len(another_answer_phrase)
        extracted_answer = row[2]["content"][option_idx]

        chat_completion.append({"role": "user", "content": medqa_full_prompt(extracted_question_and_options)["full_prompt"]})
        chat_completion.append({"role": "assistant", "content": f"### Explanation:\n{row[2]['content']}\n### Answer: {extracted_answer}"})
        return chat_completion, extracted_answer

    def _process_meerkat_instruction(self, row: list[dict[str, str]]) -> list[dict[str, str]]:
        chat_completion = list()
        chat_completion.append({"role": "system", "content": f"{SYSTEM_PROMPT}"})
        if row[1]["content"]:
            chat_completion.append({"role": "user", "content": row[1]["content"]})
        else:
            chat_completion.append({"role": "user", "content": row[0]["content"]})
        chat_completion.append({"role": "assistant", "content": row[2]["content"]})
        if len(row) > 3:
            for i in range(3, len(row)):
                chat_completion.append({"role": "user", "content": row[i]["content"]})
        return chat_completion
