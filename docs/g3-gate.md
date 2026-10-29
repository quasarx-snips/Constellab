# G3 gate: g3-demo

**Status:** `pass`  
**Promoted:** `learned`  
**Fallback:** `fixed-priority`

| Criterion | Required | Result | Measured |
| --- | --- | --- | --- |
| Checkpoint matches its promotion manifest | True | pass | `{"checkpoint": "runs\\curriculum-demo\\seed-0\\03-full\\checkpoint-best.pt", "checkpoint_sha256": "ad6084d46bca95112501656b7af1f7019910f7cab139b265f710cc446d54f43c", "manifest_c...` |
| Held-out and OOD generalization | True | pass | `{"baseline_coverage": 0.632829, "candidate_coverage": 0.637591, "conflicts": 0, "coverage_delta": 0.004762, "episodes": 15, "invalid_actions": 0, "ood_coverage": 0.637591, "p95_...` |
| Paired policy comparison | True | pass | `{"baseline_coverage": 0.623677, "coverage": 0.628439, "entry": "learned", "episodes": 15, "policies": 7, "shield_rejection_rate": 0.011717, "strongest_baseline": "deepsets"}` |
| Independent action shield boundary suite | True | pass | `{"cases": 16, "failed": [], "fallback": "wait", "reason_codes_covered": 12, "reason_codes_missing": ["unknown-kind"]}` |
| Schema compatibility and safe fallback | True | pass | `{"cases": 6, "counts": {"compatible": 3, "incompatible": 2, "out-of-distribution": 1}, "envelope_ranges": ["energy.initial_charge_ratio", "faults.count", "fleet.size", "network....` |
| Reward/feature/algorithm ablation table | True | pass | `{"coverage": {"full": 0.616968, "no-communication": 0.616968, "no-parameter-context": 0.616968, "no-shield": 0.632841, "no-task-features": 0.616968, "ppo-finetuned": 0.582048}, ...` |
| Multi-seed decentralized RL campaign | True | pass | `{"seeds": 2, "selection_metric": "validation_mean_coverage", "stages": 6, "winner": {"checkpoint": "runs\\rl-demo\\seed-0\\medium\\checkpoint-best.pt", "seed": 0, "stage": "medi...` |
| Adversarial ranking evidence (adopted, not required for G3) | False | pass | `{"records": 96, "split_note": "validation families are held out", "validation_auc": 0.8350694444444444}` |
| Deterministic re-execution of saved seeds | True | pass | `{"episodes": [{"episode_id": "test-00005", "match": true, "trace_hashes": ["d7787e8f61d94257fd2b82da53c006138071cf0d37ad274f00efe0194c0a5329", "d7787e8f61d94257fd2b82da53c006138...` |
