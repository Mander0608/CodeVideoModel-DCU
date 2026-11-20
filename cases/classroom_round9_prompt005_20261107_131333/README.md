# User-requested round9 prompt-only control

Compare with original attempt001, not masked attempts002-004. Same raw partial Blender video, one i8 image, seed42,20steps,keep5 and runner. Only effective model prompt changes. Both prompts and unified diff retained. No new image, explicit mask, pasteback or scene correction.

New wording describes the returning left known strip, interior gray gap, nonphysical gray boundaries, stationary existing board/furniture, and exact source timing. The requested known-pixel retention is textual and is not guaranteed by the original sampler.

Serial priority after active round10 masked002 ends, before launching round11. Check resources and deploy with deploy_round9_prompt005.py once; that script leaves the main chain's current run unchanged and tracks this side experiment separately. Preserve/audit round10 normally. Never rerun prepare_round9_prompt005.py.

After completion: run audit_cvm_attempt.py --round 9 --attempt attempt_005_prompt_only. Compute local motion-chain diagnostic when direct endpoint correspondence is unreliable; inspect input/output and compare with attempt001. Archive actual prompt/config/environment/timing/logs/output/hash. Report whether completion, motion and structural retention improved separately. Do not automatically replace corrected i9 or reconstruct already-running downstream rounds from this retrospective test.
