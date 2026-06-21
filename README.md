# SLM 연구 코드 최적화 (고급 파이썬 프로그래밍 기말과제)

본 저장소는 **vLLM 으로 오픈소스 LLM 을 의료 벤치마크에서 평가**하는 연구 파이프라인
**SLM** 의
두 모듈을 대상으로, 수업에서 학습한 **자료구조 / 제너레이터·lazy evaluation / 클래스 설계(SRP)·dataclass /
데코레이터 / 벤치마크 및 최적화** 기법을 적용해 성능과 구조를 개선한 결과물이다.

- 학번/이름: **202550167 현준서**
- 최적화 대상 (최적화 전 = `src/before/`, 후 = `src/after/`)
  1. **데이터 전처리 파이프라인** — `src/before/data_pipeline/{data_saver.py, utils.py}` → `src/after/data_pipeline/`
  2. **vLLM 기반 Evaluator** — `src/before/evaluation/{evaluator_core.py, prompt_engineer.py}` → `src/after/evaluation/`

> 처리 성능은 데이터 내용이 아니라 입력 크기·스키마에 좌우되므로, 입력 크기 N 을 정밀히 바꾸고 결과를
> 재현하도록 실제 벤치마크와 **동일 스키마의 synthetic data** 와 결정적 **FakeBackend** 로 측정한다.
> 따라서 `torch`/`vllm` 없이 CPU 만으로 재현 가능하다. 합성이 실제 데이터를 대표하는지는
> `bench_synth_vs_real.py`(실제 MedQA 와 비교)로 별도 검증한다.

## 디렉터리 구조

```
SLM-optimization/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ common/chat_templates.py        # 공유 순수 함수(원본 vendor)
│  ├─ before/                         # ── 최적화 전(baseline) ──
│  │  ├─ data_pipeline/{utils.py, data_saver.py}
│  │  └─ evaluation/{evaluator_core.py, prompt_engineer.py}
│  └─ after/                          # ── 최적화 후 ──
│     ├─ data_pipeline/{utils.py, data_saver.py, registry.py, decorators.py}
│     └─ evaluation/{config.py, backends.py, request_builder.py,
│                    aggregator.py, evaluator.py, decorators.py}
├─ benchmark/
│  ├─ make_synthetic_data.py          # 전처리용 CSV/JSONL 생성
│  ├─ make_synthetic_responses.py     # evaluator용 요청 + FakeBackend
│  ├─ run_benchmark_pipeline.py       # 전처리 before/after 측정
│  ├─ run_benchmark_evaluator.py      # evaluator before/after 측정
│  ├─ run_benchmark_micro.py          # 집계/반복본/재시도 미세 최적화 격리 측정
│  ├─ bench_decorator_overhead.py     # 데코레이터 호출당 오버헤드 측정
│  ├─ bench_synth_vs_real.py          # 합성 vs 실제 데이터(MedQA) 대표성 비교
│  ├─ bench_pe_compare.py             # 실제 벤치마크를 P코어/E코어 고정 비교
│  └─ verify_gpu.py                   # 실제 GPU 모델(Qwen3-0.6B) 실증
├─ tests/
│  └─ test_equivalence.py             # before==after / wraps 보존 검증
└─ results/
   ├─ pipeline_results.csv, evaluator_results.csv, decorator_overhead.csv
   ├─ micro_aggregate.csv, micro_repeat.csv, micro_retry.csv, synth_vs_real.csv
   └─ figures/*.png
```

동일 함수의 최적화 전/후 구현이 `before/` 와 `after/` **폴더에 나란히** 위치하므로, 두 디렉터리를
비교하면 변경 내용을 한눈에 확인할 수 있다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 정확성·재현성 테스트 (before==after, functools.wraps 보존 등) — pytest 불필요
python tests/test_equivalence.py     # 4/4 통과

# 2) 벤치마크 (CSV + 그래프 자동 생성, before/after 출력 동일성 검증 포함)
python benchmark/run_benchmark_pipeline.py
python benchmark/run_benchmark_evaluator.py
python benchmark/run_benchmark_micro.py         # 집계/반복본/재시도 미세 최적화 격리
python benchmark/bench_decorator_overhead.py    # 데코레이터 호출당 오버헤드(ns)
python benchmark/bench_pe_compare.py            # 실제 벤치마크를 P코어/E코어 고정 비교(psutil 필요)
python benchmark/bench_synth_vs_real.py         # 합성 vs 실제 데이터(MedQA) — datasets·인터넷 필요

# 3) (선택) 실제 GPU 모델 실증 — CUDA GPU + torch/transformers 필요
python benchmark/verify_gpu.py       # Qwen3-0.6B 를 GPU 에 올려 추론 + Evaluator E2E
```

### GPU 실증 (카테고리 E)

`benchmark/verify_gpu.py` 는 재설계한 `GenerationBackend` Protocol 이 `FakeBackend` 뿐 아니라
**실제 GPU 모델**에서도 동작함을 보인다. 동일 인터페이스의 `TransformersBackend`(`src/after/evaluation/backends.py`)로
교체하는 것만으로 **Evaluator 코드를 바꾸지 않고** 로컬 GPU(RTX 3060) 에서 Qwen3-0.6B 추론을 수행하고,
의료 MCQ 평가를 end-to-end 로 실행한다(예측 = 정답, accuracy=1.0). 네이티브 Windows 에서 vLLM 은 미지원이라
transformers 로 시연하며, Linux/WSL 에서는 같은 Protocol 의 vLLMBackend 로 그대로 대체된다.

각 벤치마크는 실행 시간(평균±표준편차, warm-up 1 + 7회)과 `tracemalloc` peak memory 를 입력 크기별로
측정하고, **before 와 after 의 출력이 바이트 단위로 동일한지(JSONL 해시 / final answer 시퀀스)** 를
`assert` 로 검증한다 — 즉 최적화는 결과를 바꾸지 않고 구조·성능만 개선한다.

## 적용한 최적화 요약

| 카테고리 | 전처리 | Evaluator |
|---|---|---|
| A 자료구조·복잡도 | `match`→dict Registry, `.find` 1회화, 정규식 패턴 1회 생성·재사용 | `pop(0)` O(n²)→`deque` O(n), `mode`→`Counter` |
| B 제너레이터·lazy | `read_csv`/`readlines`→`csv.DictReader`/스트리밍 파이프라인 | `extend([req]*n)`→`itertools`, `itertools.batched` |
| C 클래스·SRP | Reader/Transformer/Writer 분리, `ProcessorConfig`(frozen), `slots`/`frozen` dataclass, `Protocol` | God-method 분해, `GenerationBackend` Protocol(vLLM↔Fake 교체), `EvalConfig` |
| D 데코레이터 | `@timed`/`@logged`/`@validate_path` | `@retry`(지수 백오프)/`@timed`/`@lru_cache` |

자세한 진단·근거·결과 해석은 과제 제출물로 별도 제공하는 보고서 PDF 를 참조.

## 핵심 변경 (before → after)

전체 코드 대신 각 카테고리의 **지배 연산이 바뀐 핵심 줄**만 발췌한다.

### B. 제너레이터 스트리밍 (전처리) — peak memory `O(N)` → `O(1)`

```python
# before: 전체 DataFrame 적재 + iterrows + 리스트 누적 후 일괄 저장
df = pd.read_csv(path)
processed = []
for _, row in df.iterrows():
    processed.append(SupervisedData(korean_default_prompt(row.to_dict(), ...)))
save_jsonl(processed, out)             # 중간 리스트 2~3개 동시 보유, peak O(N)
```

```python
# after: csv.DictReader 제너레이터 → 변환 → 스트리밍 기록 (중간 리스트 0)
items = (SupervisedData(korean_default_prompt(row, ...))
         for row in iter_csv_rows(path))           # lazy
save_jsonl((to_record(it) for it in items), out)   # 한 줄씩 기록, peak O(1)
```

### A. `deque` 로 자료구조 교체 (Evaluator budget forcing) — `O(n²)` → `O(n)`

```python
# before: list.pop(0) 을 루프에서 호출 → 매번 O(n), 전체 O(n^2)
token_ids = [o.outputs[0].token_ids for o in longer_outputs]
for idx, remained in enumerate(budget_remained):
    if remained:
        remained_budgets[idx] -= len(token_ids.pop(0))
```

```python
# after: collections.deque.popleft 으로 O(1) → 전체 O(n)
queue = deque(o.outputs[0].token_ids for o in longer_outputs)
for idx, remained in enumerate(budget_remained):
    if remained:
        remained_budgets[idx] -= len(queue.popleft())
```

### C. `Protocol` 의존성 역전 (Evaluator) — vLLM 하드코딩 → 백엔드 교체 가능

```python
# before: vLLM 을 클래스에 하드코딩 → GPU 없이는 단위 테스트 불가
class Evaluator:
    def __init__(self, ...):
        self.vllm = LLM(model=..., tensor_parallel_size=2)
```

```python
# after: GenerationBackend Protocol 로 주입 (의존성 역전)
class GenerationBackend(Protocol):
    def chat(self, requests) -> list: ...
    def generate(self, prompts) -> list: ...

evaluator = Evaluator(backend, EvalConfig(...))   # Fake ↔ vLLM ↔ Transformers 교체
# 추출·voting·budget 로직은 backend 와 무관하게 동일 → FakeBackend 로 CPU 테스트, TransformersBackend 로 실 GPU 추론
```

## 비고

- 원본 SLM 저장소의 전체 코드/데이터는 포함하지 않으며, 최적화 대상 모듈의 before/after 만 재현 형태로 담는다.
- 보고서(PDF)는 과제 제출물로 별도 제공하며, 이 저장소에는 코드·벤치마크·결과만 포함한다.
