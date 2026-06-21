"""AFTER: 생성 백엔드 추상화 (카테고리 C — 의존성 역전).

`GenerationBackend` Protocol 로 "텍스트 생성" 인터페이스를 명시한다.
실서비스의 `vLLMBackend` 와 테스트/벤치마크용 `FakeBackend` 를 **교체 가능**하게 만들어,
GPU 없이도 host-side orchestration(요청 반복·budget bookkeeping·추출·voting)을
단위 테스트하고 벤치마크할 수 있게 한다. 이것이 원본 evaluator 의 가장 큰 구조적 한계
(vLLM 하드코딩으로 인한 테스트 불가)를 해소한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationBackend(Protocol):
    """vLLM `LLM` 과 동일한 표면을 갖는 생성 백엔드."""
    def chat(self, requests: list[list[dict[str, str]]]) -> list: ...
    def generate(self, prompts: list[str]) -> list: ...
    def get_tokenizer(self): ...


class ResultSink(Protocol):
    """결과 기록 싱크(wandb 등). 평가 로직과 분리한다."""
    def log_results(self, data_name: str, resp_text_list, filtered_resps_list, final_resps) -> None: ...


class NullSink:
    """아무것도 하지 않는 기본 싱크(벤치마크/테스트용)."""
    def log_results(self, data_name: str, resp_text_list, filtered_resps_list, final_resps) -> None:
        return None


# --- 실제 GPU 백엔드 (의존성 역전의 실증) ------------------------------------
class _Gen:
    __slots__ = ("text", "token_ids")

    def __init__(self, text: str, token_ids):
        self.text = text
        self.token_ids = token_ids


class _Out:
    __slots__ = ("outputs",)

    def __init__(self, gen: _Gen):
        self.outputs = [gen]


class TransformersBackend:
    """HuggingFace transformers 기반 실제 생성 백엔드 (GenerationBackend Protocol 충족).

    `FakeBackend` 와 **동일한 인터페이스**(chat/generate/get_tokenizer)를 구현하므로,
    Evaluator 코드를 한 줄도 바꾸지 않고 GPU 모델로 교체할 수 있다.
    이것이 원본의 vLLM 하드코딩(테스트 불가)을 해소한 Protocol 설계의 실증이다.

    (네이티브 Windows 에서 vLLM 은 미지원이므로 transformers 로 시연하며, Linux/WSL 에서는
     동일한 Protocol 을 따르는 vLLMBackend 로 그대로 대체 가능하다.)
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", device: str = "cuda",
                 max_new_tokens: int = 256, dtype: str = "float16"):
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device)
        self.device = device

    def _generate_one(self, prompt_text: str) -> _Out:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return _Out(_Gen(text, new_ids.tolist()))

    def chat(self, requests: list[list[dict[str, str]]]) -> list:
        outs = []
        for messages in requests:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outs.append(self._generate_one(text))
        return outs

    def generate(self, prompts: list[str]) -> list:
        return [self._generate_one(p) for p in prompts]

    def get_tokenizer(self):
        return self.tokenizer
