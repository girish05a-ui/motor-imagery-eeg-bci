"""CPU smoke checks for the EEGNet input/output contract."""

import torch

from src.models.eegnet import EEGNet


def _assert_finite(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), "Tensor contains NaN or Inf"


def test_eegnet_cpu_forward_backward_and_eval() -> None:
    torch.manual_seed(7)
    model = EEGNet()
    criterion = torch.nn.CrossEntropyLoss()

    for batch_size in (1, 2):
        model.train()
        x = torch.randn(batch_size, 1, 22, 750)
        y = torch.arange(batch_size, dtype=torch.long) % 4
        logits = model(x)
        assert logits.shape == (batch_size, 4)
        _assert_finite(logits)

        loss = criterion(logits, y)
        _assert_finite(loss)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                _assert_finite(parameter.grad)

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 22, 750))
    assert logits.shape == (2, 4)
    _assert_finite(logits)
