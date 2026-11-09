# Experiment Log

## 2026-10-27 — Stage 1 audit and Stage 2 EEGNet smoke test

- Verified `data/processed/A01-A09T_pooled_split_data.npz` is a real 740,961,121-byte NPZ archive, not a Git LFS pointer. It contains finite A01–A09 T-session arrays, 22 channels, 2,592 total trials, and disjoint within-subject partitions.
- Confirmed the existing pooled partition has every participant in train, validation, and test; it is not valid for unseen-participant evaluation.
- Confirmed `data/raw/` and GDF recordings are absent. Processed artifacts are sufficient for the approved cross-subject split; raw-data regeneration is not available locally.
- Added `src/models/eegnet.py` and `tests/test_eegnet_smoke.py`. CPU smoke command passed: `python -m pytest tests/test_eegnet_smoke.py -q` (`1 passed`). PyTorch 2.8.0+cpu; CUDA unavailable.
- `python -u src/check_pooled_data.py` verified pooled keys/shapes/finiteness/disjoint indices. `python -u src/validate_handoff.py` passed 10/10 checks for the legacy A01 archive only.

## 2026-10-27 — Stage 3 cross-subject data contract started

- Approved protocol: A01–A07 development only; A08–A09 final test only. A separately named archive will be generated without changing the existing pooled archive.
- Planned development split: reproducible subject-and-class stratified 75/25 split over all 2,016 A01–A07 trials (1,512 train, 504 validation). Normalization will be fitted solely on the 1,512 training trials and frozen for validation and A08–A09 test.
- No model training, test-set model selection, or CNN+LSTM work has been performed.

## 2026-10-27 — Stage 3 cross-subject artifact and loader completed

- Added `src/create_cross_subject_split.py`. It reconstructs the 2,592 source trials using pooled global indices, preserves the original pooled archive, then writes `data/processed/A01-A07T_dev_A08-A09T_test_cross_subject_data.npz` and its JSON metadata.
- New protocol: A01–A07 are stratified by participant-and-class with seed 42 into 1,512 training and 504 validation trials; all A08/A09 trials form the 576-trial final test set. Each development participant contributes 216/72 train/validation trials; A08/A09 contribute 288 each to test. Labels remain 0–3 and channel order/window remain unchanged.
- Training-only z-score parameters are stored in the new artifact and applied unchanged to validation/test. No A08/A09 samples contribute to the fit.
- Extended `src/load_data.py` with explicit `dataset='a01'` (default/backward compatible), `dataset='pooled'`, and `dataset='cross_subject'` selection. Existing A01 behavior remains the default.
- Added `tests/test_cross_subject_data_contract.py` for legacy compatibility, pooled loader availability, participant separation, source-index separation, shapes, labels/class counts, channel order, and normalization.
- Generator initially exceeded the foreground terminal's 30-second limit during compression and left a partial output. The completion was allowed to finish in a hidden process; final archive and metadata are present. No source archive was modified.
- Completed test commands: `python -m pytest tests/test_cross_subject_data_contract.py::test_legacy_a01_loader_remains_backward_compatible -q` (`1 passed in 17.24s`); `...::test_pooled_loader_is_available_without_changing_legacy_default -q` (`1 passed in 14.67s`); `...::test_cross_subject_participant_separation_shapes_labels_and_normalization -q` (`1 passed in 14.92s`); `...::test_cross_subject_loader_uses_the_separate_artifact -q` (`1 passed in 14.60s`). A combined suite was not used as pass evidence because the terminal time limit interrupted it.
- Completed `python -m pytest tests/test_cross_subject_data_contract.py::test_cross_subject_test_uses_frozen_training_statistics -q` (`1 passed in 18.10s`): the entire A08/A09 normalized test tensor exactly equals raw test data transformed by the stored A01–A07 training mean/std.
- Completed `python -m py_compile src/create_cross_subject_split.py src/load_data.py src/models/eegnet.py` successfully.
- Current status: Stage 3 data contract is implemented and verified; no training has started. Await explicit approval before training EEGNet.

## 2026-10-27 — EEGNet baseline launched

- Added `src/train_eegnet.py` for the fixed approved CPU baseline: seed 42, batch size 32, Adam lr 1e-3, weight decay 1e-4, max 150 epochs, patience 20. It selects only on A01-A07 validation macro-F1 and evaluates A08/A09 once afterwards.
- Syntax compilation passed. Training output is directed to `results/eegnet_training.out.log` and `results/eegnet_training.err.log`; artifacts target `results/eegnet_cross_subject_baseline_seed42/`.

## 2026-11-02 — EEGNet cross-subject baseline completed

- Training command: `python -u -m src.train_eegnet --output-dir results/eegnet_cross_subject_baseline_seed42 --resume`.
- The initial direct-script invocation failed before any epoch because `python src/train_eegnet.py` cannot import the `src` package. It was corrected to module execution. No data or test metrics were produced by the failed invocation.
- The foreground terminal periodically detached from the CPU run; resume support and per-epoch `last.pt` saves were added. The completed run selected the checkpoint strictly by validation macro-F1 from A01-A07.
- Completed at epoch 134 via configured early stopping (patience 20). Best validation epoch: 114; validation accuracy: 0.519841; validation macro-F1: 0.517995.
- One final held-out evaluation then ran on A08-A09: combined accuracy 0.560764, macro-F1 0.552870. Per-subject: A08 accuracy 0.607639 / macro-F1 0.593438; A09 accuracy 0.513889 / macro-F1 0.507095.
- Saved artifacts: `config.json`, `history.jsonl`, `history.json`, `best_validation.pt`, `last.pt`, `metrics.json`, and `test_predictions.npz` in `results/eegnet_cross_subject_baseline_seed42/`.
- Important logging defect: `history.jsonl` contains 200 rows with 66 duplicate epoch numbers and out-of-order entries because an earlier resume used a stale `last.pt`. Final reported results are taken from `metrics.json` and `best_validation.pt`, not inferred from row order. Future resumes must derive their next epoch from the highest recorded epoch and avoid appending duplicate history rows.
- Next required gate: review this EEGNet baseline and approve or reject a small validation-only Stage 4 tuning plan. Do not use A08/A09 during tuning.

## 2026-11-02 — Stage 4 validation-only tuning started

- User approved Stage 4. Updated `src/train_eegnet.py` with explicit `--validation-only` mode and configurable EEGNet capacity/optimizer arguments. This mode writes validation artifacts but returns before loading A08/A09.
- Candidate C1 launched: `python -u -m src.train_eegnet --output-dir results/eegnet_tune_c1_f16_d2 --validation-only --temporal-filters 16`. It changes only temporal filter count from 8 to 16; depth multiplier, dropout, seed, learning rate, weight decay, split, channels, labels, and normalization are unchanged.
- Candidate C1 result is pending. No held-out test evaluation is authorized or performed during tuning.

- C1 launch was rechecked and is now confirmed active. At epoch 8, validation macro-F1 is 0.370726; this is an interim value only, not a model-selection result. `results/eegnet_tune_c1_f16_d2/` contains `config.json`, `history.jsonl`, `last.pt`, and `best_validation.pt`. No test predictions or test metrics were created.

- C1 completed by early stopping at epoch 65. Best epoch 45: validation accuracy 0.466270, macro-F1 0.463580. This is lower than the baseline validation macro-F1 0.517995, so C1 is rejected. A08/A09 were not loaded or evaluated.
- Rationale for C2: C1 shows that increasing temporal capacity harms validation generalization. C2 will retain the baseline architecture (8 temporal filters, depth multiplier 2) and reduce dropout from 0.50 to 0.25 to test over-regularization. It is validation-only and will use the same seed, learning rate, weight decay, data split, and normalization.
- Added `results/experiment_summary.csv` as the Excel-compatible experiment table; it records completed and future experiment results.

- C2 completed by early stopping at epoch 50. Best epoch 30: validation accuracy 0.470238, macro-F1 0.463802. Lower dropout did not improve the baseline, so C2 is rejected. A08/A09 were not loaded or evaluated.
- Candidate C3 rationale: baseline validation measurements are noisy late in training, while both capacity and reduced regularization worsened generalization. C3 retains the original EEGNet architecture and dropout but lowers Adam learning rate from 0.001 to 0.0005. It remains validation-only.

- C3 completed by early stopping at epoch 108. Best epoch 88: validation accuracy 0.484127, macro-F1 0.480497. Lower learning rate did not improve the baseline, so C3 is rejected. A08/A09 were not loaded or evaluated.
- Candidate C4 rationale: increased capacity, reduced dropout, and lower learning rate were all worse. C4 tests reduced spatial-filter capacity only (depth multiplier 1 instead of 2), retaining the baseline temporal filters, dropout, optimizer, seed, split, and normalization. It is the final planned validation-only candidate; any final-test evaluation will use only the selected validation winner.

- C4 completed by early stopping at epoch 61. Best epoch 41: validation accuracy 0.460317, macro-F1 0.448312. C4 is rejected. Increasing patience would not have helped this candidate: each of its epochs 42–61 had lower macro-F1 than epoch 41. A08/A09 were not loaded or evaluated.
- Candidate C5 rationale: rather than extending a plateau at a fixed rate, C5 uses `ReduceLROnPlateau` on validation macro-F1 (factor 0.5 after 8 non-improving epochs) with baseline architecture and hyperparameters. It remains validation-only; max epochs are 150 and early stopping patience is extended to 30 so rate reductions can take effect.

- C5 completed by early stopping at epoch 148. Best epoch 118: validation accuracy 0.500000, macro-F1 0.497763. It outperformed C1–C4 but not the baseline, so C5 is rejected. The learning rate changed from 0.001 to 0.0005 after epoch 72 because `ReduceLROnPlateau` observed 8 epochs without a new best validation macro-F1; later plateau reductions followed the same factor-0.5 rule. A08/A09 were not loaded or evaluated.
- Candidate C6 rationale: all capacity, dropout, rate, and scheduler variants remain below baseline. C6 retains the baseline architecture and optimizer but introduces small training-only label smoothing (0.05) to reduce overconfident class predictions and potentially improve cross-subject generalization. It is validation-only and does not alter labels, preprocessing, normalization, or participant split.

- C6 completed by early stopping at epoch 115. Best epoch 95: validation accuracy 0.503968, macro-F1 0.499798. It improved over C1–C5 but not the baseline, so C6 is rejected. A08/A09 were not loaded or evaluated.
- Candidate C7 rationale: hyperparameter and loss regularization changes failed to beat the baseline. C7 replaces BatchNorm with GroupNorm inside EEGNet while preserving data, split, architecture capacity, optimizer, and seed. GroupNorm does not depend on mini-batch statistics and may improve robustness to cross-subject amplitude/distribution shift. It is validation-only.

- C7 start verified on 2026-11-09. `python -m py_compile src/models/eegnet.py src/train_eegnet.py` and a GroupNorm batch-2 forward/loss/backward smoke check passed (2,932 parameters). Command: `python -u -m src.train_eegnet --output-dir results/eegnet_tune_c7_groupnorm --validation-only --normalization group`. Epoch 1 was written to the C7 history; no A08/A09 data is loaded in validation-only mode.

- C7 completed by early stopping at epoch 66. Best epoch 46: validation accuracy 0.420635, macro-F1 0.413799. GroupNorm materially underperformed the BatchNorm baseline and is rejected. A08/A09 were not loaded or evaluated.
- Stage 4 conclusion: seven controlled validation-only alternatives (higher/lower capacity, lower dropout, lower LR, plateau schedule, label smoothing, and GroupNorm) all scored below the baseline validation macro-F1 0.517995. The selected EEGNet remains the Stage 3 baseline; its one authorized held-out A08/A09 evaluation is already recorded. Stop tuning rather than perform an unbounded search. Next plan gate: approve Stage 5 CNN+LSTM implementation and controlled comparison using the same split/protocol.

## 2026-11-09 — Baseline handoff package and results organization

- Created `results/experiment_catalog.csv`, a detailed Excel-compatible table covering each EEGNet experiment's protocol, input contract, normalization, architecture, optimization, validation result, held-out-test status, and selection outcome.
- Created `results/selected_eegnet_baseline/` as the stable application-integration bundle. It contains the selected best-validation checkpoint renamed to `model_weights.pt`, config, authoritative metrics, held-out predictions, history, and an inference README. Bundle smoke test passed: checkpoint epoch 114 loaded into default BatchNorm EEGNet and produced finite `(1, 4)` logits. SHA-256: `273DA47614592FDC47E26981E9AC8AEF5E1D873AD63B6AC38832AE396F6EF12A`.
- Audited repository structure. Retained all source scripts, notebooks, data, metrics, checkpoints, and figures. Removed only generated Python `__pycache__` directories and seven verified zero-byte temporary logs. No experiment evidence or model/data artifact was removed.
- Stage 5 remains pending explicit approval; no CNN+LSTM implementation or training has begun.
