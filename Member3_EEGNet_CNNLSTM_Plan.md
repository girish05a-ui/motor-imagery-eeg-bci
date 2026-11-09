# Member 3 Implementation Plan — Motor Imagery EEG Classification
## EEGNet first, compact CNN+LSTM second

**Purpose:** This is the implementation contract for Antigravity while working in the existing `motor-imagery-eeg-bci` repository. Follow the stages in order. This plan is tailored to the updated repository supplied on 26 October 2026.

**Project:** Motor Imagery EEG Classification and Simulated Brain–Computer Interface
**Dataset:** BCI Competition IV Dataset 2a; 9 participants, 22 EEG channels, 250 Hz, four imagery classes
**Member 3 responsibility:** Deep-learning models, training, validation, checkpoints, evaluation and model comparison
**Immediate scope:** Phase 1 offline classification only. Do not build the Phase 2 simulator, frontend, backend, live-streaming system or headset integration in this task.

---

## 0. Rules that must be followed

1. **Audit the current repository before editing.** Report verified facts and blockers before making changes.
2. Work in small stages. Stop at every approval gate listed in this plan.
3. Do not rewrite Member 1's preprocessing or Member 2's CSP+LDA baseline unless a concrete defect blocks the model. Explain the evidence and ask before cross-member changes.
4. Do not assume the processed `.npz` files in a downloaded ZIP contain actual NumPy arrays. The updated ZIP may contain Git LFS pointer text files; check this before trying to load them.
5. Preserve the shared labels exactly:
   - `0`: Left Hand
   - `1`: Right Hand
   - `2`: Both Feet
   - `3`: Tongue
6. Expected model input after central windowing: 22 EEG channels × 750 time samples (3.0 seconds at 250 Hz). EEGNet will typically receive `(N, 1, 22, 750)` after adding a singleton model-input dimension. Verify actual loader behavior before implementing.
7. Use the same preprocessing, label mapping and agreed evaluation protocol for EEGNet, CNN+LSTM and CSP+LDA. Do not compare scores from incompatible splits.
8. Do not tune on the final test set. Use validation data for checkpoint selection and hyperparameter tuning.
9. Never fabricate results, pretend a command ran, or report a model as trained without logs and artifacts.
10. Keep raw recordings, large binary data, checkpoints and secrets out of Git. Inspect `.gitignore` and `.gitattributes`.
11. Avoid dumping entire notebooks, NPZ arrays or long source files into context. Inspect targeted code and summarize.
12. Do not install unrelated packages, create unnecessary abstractions, refactor unrelated code, or run broad hyperparameter sweeps without approval.
13. Append completed runs and meaningful failures to `results/EXPERIMENT_LOG.md`, with machine-readable configs/metrics/history.
14. Keep all outputs reproducible: record seed, data version, split protocol, config, package versions and checkpoint path.
15. If a required input is missing or a check fails, stop at that blocker and explain the smallest action needed.

---

## 1. Updated-repository audit — Stage 1, no edits

### 1.1 Inspect only relevant files
Inspect:
- `README.md`
- `requirements.txt`
- `.gitignore` and `.gitattributes`
- `src/load_data.py`
- `src/split_data.py`
- `src/preprocess.py`
- `src/extract_epochs.py`
- `src/process_all_subjects.py`
- `src/validate_handoff.py`
- `src/check_pooled_data.py`
- `data/processed/README.md`
- `data/processed/A01-A09T_pooled_split_metadata.json`
- existing model/training/test files, if any

Inspect notebooks only when a specific implementation detail cannot be confirmed from Python source or metadata.

### 1.2 Confirm actual file availability
For every `.npz` required by the loader:
- Check file size and first bytes.
- A file beginning with `version https://git-lfs.github.com/spec/v1` is a Git LFS pointer, **not** a usable NPZ array.
- Do not use `np.load(..., allow_pickle=True)` to try to fix a Git LFS pointer.
- Report which required binary files are real arrays versus LFS pointers/missing.
- Check whether `data/raw/` and the needed `.gdf` files actually exist. Do not infer their presence from README statements.

If the real pooled NPZ is absent or replaced by an LFS pointer, do not begin training. Ask the user to restore/download the actual artifact or regenerate it from verified raw data using the existing pipeline.

### 1.3 Confirm pooled dataset coverage
The updated repository intends to provide:
- Per-subject T-session epoch/filtered artifacts for A01T–A09T.
- `data/processed/A01-A09T_pooled_split_data.npz`
- `data/processed/A01-A09T_pooled_split_metadata.json`
- T sessions only; E sessions are intentionally excluded from this pooled supervised-training dataset.

Expected pooled dataset, to verify against real arrays rather than trust blindly:
- Total: 2,592 trials (288 per subject × 9)
- Raw epoch shape: `(2592, 22, 1000)`
- Central-window shape: `(2592, 22, 750)`
- Four classes, 648 examples per class overall
- Proposed within-subject stratified split: 1,944 train, 324 validation, 324 test
- Each split is expected to contain all nine participants: 216/36/36 trials per participant respectively
- Labels: `769→0`, `770→1`, `771→2`, `772→3`

Check these values against both the actual array archive and metadata. A JSON manifest alone is not proof that the underlying arrays are present or correct.

### 1.4 Understand the current split limitation
The pooled split metadata describes a **within-subject trial-level split**: every participant appears in train, validation and test. It can measure performance on held-out trials from participants represented during training. It does **not** demonstrate generalization to entirely unseen participants.

Do not change this split silently. First report the current split protocol, then ask for approval before designing any additional cross-subject experiment. If cross-subject generalization is a research goal, use participant-grouped holdout/cross-validation with no held-out participant trials used for training, normalization fitting, tuning or checkpoint selection.

### 1.5 Check implementation and data integrity
Run the existing handoff/pooled validation commands only if their required real data and dependencies are available. Record exact command, exit status and essential output. Verify:
- X/y shapes and dtypes
- finite values
- class distribution
- participant IDs in each split
- no trial-index overlap between train/validation/test
- channel order identical across participants
- normalization fitted on training data only, then frozen for validation/test
- central window is samples `[125:875]` at 250 Hz
- artifact masks are aligned with trials and interpreted as flags, not silently dropped
- raw, filtered, pooled arrays and metadata agree
- any train/val/test IDs or source trial provenance needed for leakage checks

If a validation command is blocked because NPZ files are Git LFS pointers, report that as a data-delivery blocker; do not claim validation passed.

### 1.6 Stage 1 report format
Return only:
1. `PASS`, `FAIL` or `BLOCKED` for dataset availability, pooled data, loader, split, normalization and dependencies.
2. Actual paths inspected.
3. Commands run and outcomes.
4. Confirmed input signature and shapes, or explain why they could not be verified.
5. Specific blockers that prevent EEGNet work.
6. Smallest proposed next steps.

**Do not edit files or train a model in Stage 1. Stop and wait for approval.**

---

## 2. Stage 2 — Environment and model smoke test, after approval

### 2.1 Environment
- Keep reusable code in Python modules/scripts; notebooks should be thin execution interfaces.
- Use local CPU for import checks, shape checks and a tiny smoke test.
- Select CUDA when available; otherwise use CPU and print that CPU is being used.
- Record Python, PyTorch, NumPy, scikit-learn versions, device, GPU name if available, seed, data archive identity and experiment ID.
- Add only the minimum required deep-learning dependencies after checking what is installed. Do not overwrite the team's dependency choices without explaining why.

### 2.2 EEGNet input/output contract
- Input data from the agreed loader: `(N, 22, 750)` if central window is selected.
- Explicitly transform to the architecture's expected layout, commonly `(N, 1, 22, 750)`.
- Output: four logits per trial, shape `(N, 4)`.
- Confirm label dtype is compatible with cross-entropy loss and labels are in `{0,1,2,3}`.

### 2.3 Required smoke tests
Before full training, verify:
- model instantiates and parameter count is reported
- forward pass works for batch sizes 1 and 2
- logits are finite and have four outputs
- loss is finite
- backward pass succeeds and gradients are finite
- evaluation mode works
- input/output shapes are covered by a small test
- the existing loader works without copying or duplicating preprocessing logic

Do not run a hyperparameter sweep at this stage.

---

## 3. Stage 3 — Establish one reproducible EEGNet baseline

Only proceed after the dataset files are real, integrity checks pass, and the split protocol is approved.

### 3.1 Training discipline
- Use the agreed loader and the existing split. Do not create a new random split inside the training script.
- Use the central 3-second window unless the team explicitly selects another window.
- Fix and record random seeds.
- Start with one conservative baseline configuration, not a large search.
- Suggested initial values to be configurable rather than hard-coded: Adam optimizer, learning rate around `1e-3`, weight decay around `1e-4`, batch size 16 or 32, max epochs around 150, early stopping patience around 20. These are starting values, not guaranteed optimal settings.
- Select the best checkpoint using validation macro-F1; document validation loss as a tie-breaker if used.
- Save both best-validation and resumable checkpoints.
- Do not evaluate the final test set repeatedly during tuning.

### 3.2 Metrics
On validation and final test, report as appropriate:
- accuracy
- macro-F1
- per-class precision, recall and F1
- confusion matrix
- per-subject metrics for the pooled within-subject experiment
- training/validation loss and metric curves
- inference latency only if measured with a documented method

State the split type clearly in every report. A pooled trial-level split must not be described as unseen-subject evaluation.

### 3.3 Experiment record
Create/update `results/EXPERIMENT_LOG.md`. For every run append:
- stable experiment ID and timestamp
- Git commit/hash if available
- data archive name/hash or version
- participant/session coverage and split protocol
- preprocessing/window/normalization
- model configuration and seed
- Python/library versions and device
- epochs completed and stopping reason
- validation metrics and final test metrics only when the final evaluation is authorized
- checkpoint, history, predictions and metrics paths
- interpretation, failures and next action

Also save JSON config and machine-readable metrics/history. Never record invented or estimated performance as a result.

---

## 4. Stage 4 — Hyperparameter tuning, after baseline review

After the baseline finishes:
1. Review learning curves, per-class performance and errors.
2. Propose a small, justified search for approval.
3. Use validation metrics only to select hyperparameters and checkpoints.
4. Avoid tuning against the test set.
5. Save every configuration and result so runs are comparable.
6. If the data/split changes, assign a new experiment ID and treat it as a new experiment family. Do not mix scores from different protocols.

Do not assume that hyperparameters tuned on A01/T remain optimal for pooled A01–A09 data. If prior A01/T tuning exists, retain it as a preliminary experiment, but establish a new baseline for the verified pooled dataset and approved split.

---

## 5. Stage 5 — CNN+LSTM, only after EEGNet is complete

Do not implement or train CNN+LSTM until:
- EEGNet smoke tests pass
- one baseline is trained reproducibly
- evaluation and artifacts are saved
- the team approves proceeding

Then implement a compact, documented CNN+LSTM with a genuine temporal sequence entering the LSTM. Use the same loader, data version, preprocessing, split protocol, model-selection rules and metrics as EEGNet. Run the same smoke tests and report a controlled comparison. A more complex model is not automatically better.

---

## 6. Remote GPU workflow — VS Code + Colab/Kaggle

### Development
- Use Antigravity as the coding agent and VS Code as the local project editor/terminal.
- Keep model/training/evaluation code in version-controlled Python files.
- Do not assume Antigravity's local terminal is executing on a remote GPU.

### Google Colab
- If using the official Google Colab VS Code extension, use a notebook as a thin remote execution interface.
- Verify `torch.cuda.is_available()` and print the GPU name before training.
- Ensure the actual repository revision and the real data files are present in the remote runtime. Do not assume local files are automatically visible remotely.
- Execute the implemented training script from a notebook cell using its documented CLI/config. Do not duplicate model logic in notebook cells.

### Kaggle fallback
- Keep scripts portable to a Kaggle GPU notebook.
- Do not assume an official direct remote kernel/API for a local VS Code terminal.
- Document provider-specific paths and dependency differences.

### Checkpoints and persistence
- Save a best-validation checkpoint and a resumable checkpoint.
- Where practical, store model, optimizer, scheduler, epoch, best metric, RNG states and complete config.
- Store artifacts on persistent storage, not only the temporary runtime filesystem.
- Load checkpoint on CPU first, then move to the active device.
- Before resuming elsewhere, verify model-code, config, dependency and dataset compatibility.
- Record interruptions, resumed epoch, checkpoint path and runtime differences.

### Resource and security rules
- Do not assume GPU quota or runtime duration is guaranteed.
- Run smoke tests before full training and prefer sequential experiments on limited compute.
- Never hard-code keys, passwords or tunnel tokens; do not expose a notebook runtime or tunnel publicly.
- Keep raw data, large checkpoints and secrets out of Git.

---

## 7. Final handoff and definition of done

Member 3's Phase 1 handoff is complete when:
- [ ] The actual pooled data archive has been verified as a real NPZ, not an LFS pointer.
- [ ] Dataset coverage and the approved split protocol are documented.
- [ ] EEGNet input/output and gradient smoke tests pass.
- [ ] A reproducible EEGNet baseline has completed.
- [ ] Validation-based model selection and final test evaluation are correctly separated.
- [ ] Checkpoints, config, history, metrics and predictions are saved.
- [ ] Experiment log contains successes and meaningful failures.
- [ ] Per-class and per-subject results are reported with limitations.
- [ ] CNN+LSTM is implemented only after EEGNet completion and uses a comparable protocol.
- [ ] No unsupported claims about unseen-subject generalization or real-time BCI are made.

**Immediate instruction:** Start with Stage 1 audit only. Do not modify files, train, tune, or implement CNN+LSTM until the audit report is reviewed and approval is given.
