# Final Pipeline Status

## Module 1 — Crack Segmentation

- Reportable completed for the current demo pipeline.
- Trajectory: r009 synthetic-only failed on real photos → r010 real-domain fine-tune → r011 repair-mask stable baseline → r012 manual-mask experimental fine-tune.
- r011 is the stable segmentation baseline.
- r012 is experimental; current evidence does not justify claiming it is clearly superior.
- `auto_r011_sensitive_low_threshold` is the high-recall demo3 mode. It is useful for the selected demo but may increase false positives and is not the global default.

## Module 2 — Inpainting

- `simple_lama` is the stable fallback backend.
- `opencv` is the classical fallback backend.
- `official_lama_pretrained` is runnable through a subprocess adapter in conda env `lama`.
- Official LaMa currently runs on local CPU, so it can be slower than `simple_lama`.
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

- Official LaMa is CPU-only in the current local setup.
- LaMa fine-tune has not been performed.
- Broader real-photo test coverage is still needed.
- LPIPS, FID, and masked-region metrics have not been measured.
- Sensitive mode may increase false positives.

## Future Work

- Build a GPU-ready official LaMa env.
- Fine-tune LaMa only after pretrained official LaMa is reviewed and a small training smoke test is designed.
- Broaden evaluation across more real old photos.
- Add LPIPS/FID/masked-region metrics if needed for the final report.
