# Final Pipeline Status

## Module 1 — Crack Segmentation

- Reportable completed for the current demo pipeline.
- Trajectory: r009 synthetic-only failed on real photos → r010 real-domain fine-tune → r011 repair-mask stable baseline → r012 manual-mask experimental fine-tune.
- r011 is the stable segmentation baseline.
- r012 is experimental; current evidence does not justify claiming it is clearly superior.
- `auto_r011_sensitive_low_threshold` is the high-recall demo3 mode. It is useful for the selected demo but may increase false positives and is not the global default.

## Module 2 — Inpainting

- `simple_lama` is the stable fallback backend.
- `simple_lama` availability can differ by runtime: it is available in the default project Python but unavailable in the Gradio venv used during fallback testing.
- If `simple_lama` is unavailable, the pipeline falls back to `opencv` and records `fallback_applied=true`, `fallback_chain=["simple_lama", "opencv"]`, and `simple_lama_reason=module_not_found`.
- `opencv` is the classical fallback backend.
- `official_lama_pretrained` is runnable through a subprocess adapter.
- GPU env `lama_gpu` has been prepared as an optional official LaMa runtime. The adapter now probes `lama_gpu` first and selects CUDA when `torch.cuda.is_available()` is true, while preserving CPU env `lama` as retry/fallback.
- If official LaMa GPU inference fails, the adapter retries with CPU env `lama` before the outer pipeline falls back to `simple_lama` or `opencv`.
- Latest GPU smoke on `demo3` records `official_lama_env_actual=lama_gpu`, `official_lama_device_actual=cuda`, `official_lama_torch_version=2.4.1+cu121`, and `official_lama_cuda_available=true`.
- Single-image timing did not show a speedup yet: GPU attempt was about 44.15s and CPU direct attempt was about 38.51s for the same demo3 input/mask. Treat GPU as available but not proven faster until broader timing is run.
- LaMa fine-tune has not been performed.

## Module 3 — Face Restoration

- CodeFormer is activated through the separate env `codeformer`.
- Metadata evidence from smoke tests records `face_restoration_applied=true` and `face_backend=codeformer` when CodeFormer is requested and succeeds.

## Demo Candidate

- Demo image: `demo3`
- Mask mode: `auto_r011_sensitive_low_threshold`
- Inpainting backend candidate: `official_lama`
- Face restoration: `codeformer_if_available`
- CodeFormer fidelity: `0.7`
- Status: pending human visual review before claiming final quality.

## Limitations

- Official LaMa GPU is available in env `lama_gpu`, but speedup is not established from the current one-image smoke test.
- LaMa fine-tune has not been performed.
- Broader real-photo test coverage is still needed.
- LPIPS, FID, and masked-region metrics have not been measured.
- Sensitive mode may increase false positives.

## Future Work

- Build a GPU-ready official LaMa env.
- Fine-tune LaMa only after pretrained official LaMa is reviewed and a small training smoke test is designed.
- Broaden evaluation across more real old photos.
- Add LPIPS/FID/masked-region metrics if needed for the final report.
