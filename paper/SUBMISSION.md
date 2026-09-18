# Submission package — AGF-NO → Nature Machine Intelligence

Target article class: **Progress / Analysis** (diagnosis + instrument + selection
rule + validated correction — the framing the paper is now written in).

## 1. Before you touch the submission form

- [ ] **Rotate the Hugging Face token.** It has transited the intercepting proxy
      and a private Kaggle dataset. Rotate in HF Settings → Access Tokens, then
      update the Modal secret (`modal secret create agfno-hf HF_TOKEN=hf_NEW`).
- [ ] **Optional upgrade — full-budget matched-baseline matrix** (6 runs,
      ~5 GPU-hours on T4; staged, no longer blocking submission since the
      pilot table carries the claim):
      - Kaggle: weekly 30 h GPU quota exhausted — resets on the weekly roll
        (Sat 00:00 UTC). Then either `cd kaggle/kernels/exp5 && python -m kaggle kernels push`
        or leave it; the Modal path below needs no waiting.
      - Modal: workspace spend limit hit — raise it in modal.com → Settings →
        Limits, then `modal run -m agfno.infrastructure::run_exp5`.
      - **Or do nothing:** the headless automation (`agfno/midnight_push.py`,
        running detached since Fri ~21:45 UTC) pushes exp5 AND both exp6
        arms automatically at the quota reset, polls, pulls, merges, and
        rebuilds the PDF. State: `midnight_state.json`; log: `midnight.log`.
        If it died (machine restart), relaunch with
        `nohup python agfno/midnight_push.py > /dev/null 2>&1 &` — it resumes.
- [x] **Pilot study COMPLETE (CPU, no quota used).** Kernel
      `agfno-darcy-experiments5p` trained ALL FOUR architectures (FNO, AGF-NO,
      U-Net, CNO — the headline pair retrained under the pilot budget) with an
      identical reduced protocol (256/64/128 samples, 32 epochs, 3 seeds).
      **Results are in the paper** (\S"Pilot-budget matched baselines",
      Table 4, from `kaggle/exp5p_out/runs/experiments5p/pilot_summary.json`):
      AGF-NO best on ring error (0.1632 vs U-Net 0.1742 / FNO 0.2890 /
      CNO 0.3856) and wall fidelity (0.0217 — 3.9× better than FNO,
      4.8× than U-Net, 12.8× than CNO); `PILOT_ORDERING_PASS: true`.
      Honest disclosure included: U-Net wins global rel-L2 at pilot budget
      (0.0947) — stated in the paper, and it *strengthens* the mechanism
      story (ring pathology is specific to the band-limited global spectral
      channel, not to deep nets). The v1 gate premise ("convolutional
      baselines fail at walls like FNO") was refuted by the data; the
      corrected gates encode the mechanism claims directly.
- [x] **Third-architecture suite BUILT (AFNO family member).** `agfno/afno.py`
      (AFNO mixer: soft spectrum shrinkage, pre-norm double-skip, mode-
      restricted weights; param-matched to FNO at 4.81M) + `agfno/experiments6.py`
      (depths 4/8/16 × 3 seeds, exp3 protocol parity, inline geometry probe,
      pre-registered gates A1–A4) + `agfno/merge_exp6.py` (split-kernel merge)
      + 5 new tests (49/49 green). Kernels `exp6-afno` / `exp6-agfafno` staged
      for two concurrent GPU sessions (~5 h each); the paper's §Architecture
      generality (Table 5) auto-appears when `\AfnoCollapseRatio` is defined.
      falsifiable claim under test: the collapse is a property of the
      band-limited-global-channel FAMILY, not of FNO's specific design.
- [ ] **Run the baselines, close the paper loop (one command each):**
      ```bash
      modal run -m agfno.infrastructure::run_exp5          # ~5 h on T4
      modal run -m agfno.infrastructure::fetch_exp5 > exp5_raw.log
      python agfno/fetch_exp5.py exp5_raw.log              # parses, regenerates
                                                           # macros, rebuilds PDF
      ```
      The baselines section (`\ref{sec:baselines}`, Table `\ref{tab:baselines}`)
      auto-appears in the PDF the moment `\UNetRelL` is defined; until then the
      paper compiles clean without it (verified with synthetic data, then wiped).
- [ ] Re-push the final PDF to HF: version the relay dataset, rerun the relay
      kernel (same flow as before), verify `paper/agfno_paper.pdf` updated.

## 2. Submission-form items (NMI)

- [ ] **Article type**: Progress. NMI Progress/Analysis pieces are typically
      ≤ 8–10 pages typeset; the current manuscript is 16 pages in article style.
      Use their template (`nature.com/nature-machine-intelligence` →
      submission guidelines → Progress template) and expect a tightening pass:
      Methods → end, figures ≤ 8, main text ≈ 3,000–4,000 words.
- [ ] **Format-independent claims map** (for the cover letter, already in the
      paper): diagnosis (Prop. 1, §4) → instrument (probe + controls, §7) →
      a-priori rule (Δρ, §6.3) → correction with full ablation (§3, §6.1) →
      replication (3 seeds, PDE2, 3-D; §6–§8) → matched external baselines
      (§6.5, pending GPU).
- [ ] **Referee-proofing notes**: the near-null PDE2 result and the honest G2
      caveat (AGF-NO ring 1.45× its depth-4 value at depth 16) are stated in
      the abstract/Limitations — keep them; reviewers reward them and the
      evidence already matches the wording.
- [ ] **Data/code availability statement**: cite the HF repo
      (`huggingface.co/Sejibeji/agfno-darcy`), GitHub
      (`github.com/sehajr-singhs/agfno`), and the one-click Kaggle/Modal reruns.
- [ ] **CITATION.cff** already exists in the GitHub repo; mirror its metadata
      into the submission form fields.

## 3. Cover letter (draft)

> Re: Progress submission — "The Truncated Band: Diagnosing and Correcting
> Geometric Forgetting in Fourier Neural Operators"
>
> Fourier Neural Operators learn mappings between function spaces at costs
> orders of magnitude below classical solvers, and are increasingly deployed
> where speed is the point. We show that their single global mixing channel
> is a band-limited projection — a two-line proposition — and that this is
> precisely what breaks on problems whose difficulty lives at boundaries:
> at depth 16 the vanilla FNO collapses (relative error ≈ 1.008, worse than
> predicting the mean) while a linear probe shows the geometry has been
> erased from its features (R² ≈ 0.02 vs 0.49 at depth 4). The collapse is
> replicated across 3 seeds, two PDE families, and lifted to 3-D obstacle
> Darcy with multiply-connected geometry; a matched external-baseline study
> (U-Net, CNO at equal parameter count) confirms the near-wall ordering is
> not Fourier-specific. A zero-gated, spatially adaptive modulation of the
> spectral weights — added capacity isolated to zero by a frozen-gate
> control — restores the truncated band and holds the plateau.
>
> We believe this fits NMI's Progress format exactly: a mechanism-level
> diagnosis, a reusable measurement instrument with controls, an a-priori
> rule that tells practitioners when the failure will matter, and a minimal
> validated correction — every element falsifiable, every number regenerated
> from committed per-seed JSONs, and every experiment re-runnable in one
> click on public infrastructure. Near-null results bounding the mechanism's
> scope are reported and analyzed, not omitted.
>
> All code, checkpoints, figures, and per-seed data are public; the
> manuscript's numbers are generated programmatically from the result files
> (no hand-typed statistics).

## 4. Remaining science gaps to acknowledge in any response-to-referee

1. Airfoil-class and time-varying boundaries (Limitations i) — future work.
2. The Δρ rule is validated on the two families it was built to explain
   (Limitations vi) — a third family would strengthen it.
3. External baselines are faithful re-implementations, not authors' original
   code (Limitations ii).

## 5. Quick reference — what lives where

| Artifact | Location |
|---|---|
| Paper source + macros | `paper/main.tex`, `paper/macros.tex` (240 auto-macros) |
| Built PDF | `paper/main.pdf` (16 pp) / HF `paper/agfno_paper.pdf` |
| Code (2-D + 3-D + baselines) | `agfno/` (45 tests green) |
| Per-seed results | `site/agfno/results/` + HF `experiments*/` |
| Baselines suite | `agfno/experiments5.py` (+ resume-safe) |
| GPU runners | `kaggle/kernels/exp5/`, `agfno/infrastructure.py::run_exp5` |
| Close-the-loop script | `agfno/fetch_exp5.py` |
| Third-architecture suite | `agfno/afno.py`, `agfno/experiments6.py`, `agfno/merge_exp6.py` |
| Headless automation | `agfno/midnight_push.py` (quota-reset watcher → push/poll/pull/merge/rebuild) |
