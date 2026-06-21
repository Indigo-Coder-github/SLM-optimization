"""로컬 GPU 사용 검증 — 소형 Qwen 모델을 실제로 GPU 에 올려 추론한다.

목적
  1) 현재 머신의 GPU(CUDA)가 실제로 동작하는지(텐서·모델이 GPU 메모리에 올라가는지) 검증.
  2) evaluator 재설계의 `GenerationBackend` Protocol 이 FakeBackend 뿐 아니라 *실제 모델* 로도
     동작함을 보여 의존성 역전 설계의 타당성을 입증(과제 카테고리 E 연계).

transformers 경로를 사용한다(네이티브 Windows 에서 vLLM 은 미지원이라 WSL/Linux 필요).
모델: Qwen/Qwen3-0.6B (없으면 Qwen/Qwen2.5-0.5B-Instruct 로 폴백).
"""

from __future__ import annotations

import sys
import time

import torch

# Windows 콘솔(cp949)에서도 유니코드 출력이 깨지지 않도록 UTF-8 로 재설정
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATES = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"]


def _mib(x: int) -> float:
    return x / (1024 ** 2)


def main():
    assert torch.cuda.is_available(), "CUDA GPU 를 사용할 수 없습니다."
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[GPU] {name} | total {_mib(total):.0f} MiB | torch {torch.__version__} | CUDA {torch.version.cuda}")

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    print(f"[GPU] baseline allocated: {_mib(base):.1f} MiB")

    model_id = None
    last_err = None
    for cand in CANDIDATES:
        try:
            print(f"[load] trying {cand} ...")
            tok = AutoTokenizer.from_pretrained(cand)
            model = AutoModelForCausalLM.from_pretrained(cand, torch_dtype=torch.float16).to(dev)
            model_id = cand
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[load] {cand} 실패: {type(e).__name__}: {e}")
    if model_id is None:
        raise RuntimeError(f"모델 로드 실패: {last_err}")

    after_load = torch.cuda.memory_allocated()
    n_params = sum(p.numel() for p in model.parameters())
    on_cuda = next(model.parameters()).is_cuda
    print(f"[load] {model_id} loaded | params {n_params/1e6:.1f}M | on_cuda={on_cuda} "
          f"| weights ~{_mib(after_load - base):.1f} MiB on GPU")

    # 의료 도메인 프롬프트(SLM 맥락)로 실제 생성
    messages = [
        {"role": "system", "content": "You are a medical expert. Answer concisely."},
        {"role": "user", "content": "고혈압 환자에게 일반적으로 권장되는 생활습관 교정 3가지를 한국어로 짧게 답하세요."},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(dev)

    # warm-up (CUDA 커널 컴파일/캐시)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    n_new = gen_ids.shape[0]
    peak = torch.cuda.max_memory_allocated()

    print("\n========== 생성 결과 ==========")
    print(gen_text.strip()[:600])
    print("================================")
    print(f"[infer] new tokens={n_new} | {dt*1000:.1f} ms | {n_new/dt:.1f} tok/s")
    print(f"[GPU] peak allocated during run: {_mib(peak):.1f} MiB")
    print("[OK] GPU 사용 검증 완료 - 모델 가중치/활성값이 GPU 메모리에 적재되어 추론 수행됨.")

    # 모델 해제 후 end-to-end 데모(GPU 메모리 회수)
    del model
    torch.cuda.empty_cache()
    return model_id


def demo_evaluator_on_gpu(model_id: str):
    """재설계된 Evaluator 를 실제 GPU 모델(TransformersBackend)로 end-to-end 실행.

    Evaluator 코드는 그대로 두고 backend 만 FakeBackend → TransformersBackend 로 교체했다.
    => GenerationBackend Protocol(의존성 역전) 설계가 실제 GPU 모델에서도 동작함을 실증.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.after.evaluation.backends import TransformersBackend
    from src.after.evaluation.config import EvalConfig
    from src.after.evaluation.evaluator import Evaluator

    # 실제로 정답이 정해진 의료 객관식 2문항(영어, 모델이 답할 수 있는 형태)
    def mcq(q, a, b, c, d, e):
        user = (f"Read the question and choose the single best option. "
                f"End your answer with 'Answer: X'.\n### Question:\n{q}\n### Options:\n"
                f"A. {a}\nB. {b}\nC. {c}\nD. {d}\nE. {e}")
        return [{"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": user}]

    messages = [
        mcq("Which electrolyte abnormality is most characteristic of primary hyperaldosteronism?",
            "Hyperkalemia", "Hypokalemia", "Hypercalcemia", "Hyponatremia", "Hypomagnesemia"),
        mcq("What is the first-line antibiotic class for uncomplicated streptococcal pharyngitis?",
            "Fluoroquinolone", "Aminoglycoside", "Penicillin", "Macrolide", "Tetracycline"),
    ]
    labels = ["B", "C"]

    print("\n[E2E] 재설계 Evaluator + TransformersBackend(실 GPU 모델) 실행 ...")
    backend = TransformersBackend(model_id=model_id, max_new_tokens=256)
    evaluator = Evaluator(backend, EvalConfig(num_repeats=1))
    acc, final = evaluator.evaluate_model(messages, labels, data_name="gpu_demo")
    print(f"[E2E] 예측={final} 정답={labels} accuracy={acc}")
    print("[E2E][OK] GenerationBackend Protocol 로 GPU 모델 교체 성공 — evaluator 코드 변경 0줄.")


if __name__ == "__main__":
    mid = main()
    demo_evaluator_on_gpu(mid)
