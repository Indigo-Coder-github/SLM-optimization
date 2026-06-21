"""공유 chat-template 함수 (원본 SLM `src/utils/chat_templates.py`에서 발췌·정리).

before/after 양쪽이 동일한 변환 로직을 사용하도록 한 곳에 모아 두었다.
모두 부수효과 없는 순수 함수이므로 최적화 대상이 아니라 *공통 의존성*으로 취급한다.
"""

SYSTEM_PROMPT = (
    "You are a medical expert.\n"
    "You should provide accurate and helpful answers to the following questions "
    "or requests based on your medical knowledge and the provided context.\n"
    "You should not provide harmful or misleading information.\n"
    "Please ensure your responses are clear and concise, adhering to the format "
    "specified in the question."
)


def korean_default_prompt(x: dict, input_column_name: str, output_column_name: str) -> list[dict[str, str]]:
    """한국어 instruction-output chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### 질문:\n{x[input_column_name]}"},
        {"role": "assistant", "content": f"### 답변:\n{x[output_column_name]}"},
    ]


def kormedmcqa_and_aihub_full_prompt(x: dict) -> dict:
    prompted_text = f'''다음 질문을 읽고 가지고 있는 지식을 최대한 활용하여 설명한 다음 가장 적절한 보기를 하나 선택하세요.
### 질문:
{x["question"]}
### 보기:
A. {x["A"]}
B. {x["B"]}
C. {x["C"]}
D. {x["D"]}
E. {x["E"]}'''
    x["full_prompt"] = prompted_text
    return x


def kormedmcqa_and_aihub_chat_template(x: dict, file_type: str) -> list[dict[str, str]]:
    """KorMedMCQA 훈련용 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": kormedmcqa_and_aihub_full_prompt(x)["full_prompt"]},
    ]
    if file_type == "test":
        return messages
    if "explanation" in x:
        messages.append({"role": "assistant", "content": f'### 설명:\n{x["explanation"]}\n### 정답: {x["answer"]}'})
    else:
        messages.append({"role": "assistant", "content": f'### 정답: {x["answer"]}'})
    return messages


def pubmedqa_chat_template(x: dict) -> list[dict[str, str]]:
    """PubMedQA chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### Question:\n{x['question']}\n### Context:\n{x['context']}"},
        {"role": "assistant", "content": f"### Explanation:\n{x['long_answer']}\n### Answer:\n{x['final_decision']}"},
    ]


def medqa_full_prompt(x: dict) -> dict:
    prompted_text = f'''Read the following question, use your knowledge to answer them, and then select the most appropriate option.
### Question:
{x["question"]}
### Options:
A. {x["options"]["A"]}
B. {x["options"]["B"]}
C. {x["options"]["C"]}
D. {x["options"]["D"]}'''
    x["full_prompt"] = prompted_text
    return x
