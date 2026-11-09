# Selected EEGNet Baseline Bundle

This folder is the stable handoff package for application integration.

- `model_weights.pt`: best-validation EEGNet checkpoint (epoch 114).
- `config.json`: fixed model/training configuration.
- `metrics.json`: validation and one authorized A08/A09 final-test result.
- `test_predictions.npz`: held-out predictions with subject IDs.
- `history.json`: training history; use `metrics.json` for the authoritative summary.

## Inference contract

- Model: `src.models.eegnet.EEGNet()` with its defaults (`BatchNorm`, 8 temporal filters, depth multiplier 2, dropout 0.50).
- Input: `float32`, shape `(N, 1, 22, 750)`.
- Data source: the `cross_subject` loader, normalized central window.
- Labels: `0=Left Hand`, `1=Right Hand`, `2=Both Feet`, `3=Tongue`.
- Preprocessing: use the frozen A01-A07 train normalization statistics from `data/processed/A01-A07T_dev_A08-A09T_test_cross_subject_data.npz`; never fit on integration inputs or test participants.

This is an offline research model, not a calibrated real-time clinical/BCI deployment model.

## Integrity

`model_weights.pt` SHA-256: `273DA47614592FDC47E26981E9AC8AEF5E1D873AD63B6AC38832AE396F6EF12A`.
