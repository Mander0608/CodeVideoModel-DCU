# User-authorized round9 natural-completion prompt006

Only this additional inference is authorized. Main chain remains stopped; no round11/12, no automatic retry/reference replacement.
Same raw Blender partial video, one i8 reference, seed42/20steps/keep5 and runner as original001 and005. Only prompt changes. Source draft is prompt_natural_completion_v006.txt.
Deploy once via deploy_round9_prompt006.py after resource check. Never rerun preparation. Collector only downloads existing results.
After completion run audit_cvm_attempt.py --round 9 --attempt attempt_006_natural_completion. Adapt diagnose_round9_motion.py to006, never overwrite original001. Compare001/005/006 completion, seams, motion and structure separately, visually inspect first/intermediate/end frames, retain all input/config/actualprompt/code/environment/timing/logs/output/SHA256. Archive and report then stop. Do not rerun hardcoded005 archivers.
