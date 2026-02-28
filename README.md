# AR-QL (Action-Robust Q-Learning) — Gymnasium MuJoCo 구현

이 저장소는 `SMC_2024_RRL_Comments.pdf`에 정리된 **AR-QL(Action-Robust Q-Learning)** 알고리즘을 Gymnasium MuJoCo 환경에서 학습/평가하기 위한 최소 구현입니다.

- **핵심 아이디어**
  - Protagonist는 별도 정책 네트워크 없이 **Q(s, a)** 에서 직접 좋은 행동 `a^P`를 샘플링합니다.
  - Adversary는 정책 `π_ϕ(s)=a^A`를 학습하며, Q값을 **낮추는 방향**으로 업데이트됩니다.
  - 실제 환경에는 혼합 행동 `ã = (1-α)a^P + α a^A (+ 탐색 노이즈)`를 적용합니다.
- Protagonist 샘플링은 2가지 방식 지원:
  - `rs`: Rejection Sampling
  - `ld`: Langevin Dynamics

---

## 파일 구조

- `AR_QL.py`  
  AR-QL 핵심 구현(Agent, Critic, Adversary policy, Protagonist sampler(RS/LD), ReplayBuffer)
- `main.py`  
  학습 스크립트(train). 환경 step → 리플레이 저장 → critic/adversary 업데이트 → 주기 평가/체크포인트 저장
- `eval.py`  
  체크포인트 로드 후 평가(평균 return). (현재 인자/옵션 일부 수정 필요 — 아래 “Known Issues” 참고)
- `utils.py`  
  seed/device/obs flatten/soft update 등 공용 유틸
- Action Robust Reinforcement Learning with Highly Expressive Policy (IEEE SMC, 2024) 구현.
  논문/주석(알고리즘 라인 매핑 참고용)
