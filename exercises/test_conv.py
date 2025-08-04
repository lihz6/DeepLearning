import pytest


def test_numpy_convolve():
    """np.convolve不支持广播，但满足交换律"""
    import numpy as np

    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    kernel = np.array([0.2, 0.4, 0.3, 0.1])
    """1. 不支持广播，signal 和 kernel 必须是 1 维的"""
    with pytest.raises(ValueError):
        np.convolve(signal.reshape(1, -1), kernel, mode="same")

    with pytest.raises(ValueError):
        np.convolve(signal.reshape(-1, 1), kernel, mode="same")

    with pytest.raises(ValueError):
        np.convolve(signal, kernel.reshape(1, -1), mode="same")

    with pytest.raises(ValueError):
        np.convolve(signal, kernel.reshape(-1, 1), mode="same")

    """
    2. 满足交换律，即 np.convolve(signal, kernel) == np.convolve(kernel, signal)

    有两个机制保证满足交换律：

    - 在计算时，np.convolve 会将 kernel 进行翻转(kernel[::-1]), 确保计算结果一致；
    - 当 signal 的长度小于 kernel 时，np.convolve 会将交换 signal 和 kernel (np.convolve(kernel, signal))，确保 mode="same" 时输出长度一致。
    """
    assert np.allclose(
        np.convolve(signal, kernel, mode="same"),
        np.convolve(kernel, signal, mode="same"),
    )

    assert np.allclose(
        np.convolve(signal[:4], kernel, mode="same"),
        np.convolve(kernel, signal[:4], mode="same"),
    )


def test_torch_conv1d():
    """
    torch.nn.functional.conv1d(
        input: (N?, C, L),
        weight: (C', C / G, K <= L),
        bias: (C') = None, /, *,
        stride: int = S,
        padding: int = P,
        dilation: int = D,
        groups: int = G
    ) -> (N?, C', (L + 2P - D(K - 1) -1) // S + 1)
    """
    import torch
    import torch.nn.functional as F

    """1. input 的 shape 必须是 (N?, C, L), weight 的 shape 必须是 (C', C, K <= L), bias 的 shape 必须是 (C')"""

    assert F.conv1d(torch.randn(2, 4, 20), torch.randn(16, 4, 3)).shape == (2, 16, 18)
    assert F.conv1d(torch.randn(4, 20), torch.randn(16, 4, 3)).shape == (16, 18)
    assert F.conv1d(torch.randn(4, 8), torch.randn(16, 4, 8)).shape == (16, 1)
    assert F.conv1d(
        torch.randn(4, 20), torch.randn(16, 4, 3), torch.randn(16)
    ).shape == (16, 18)

    with pytest.raises(
        RuntimeError, match=r"Kernel size can't be greater than actual input size"
    ):
        F.conv1d(torch.randn(4, 9), torch.randn(16, 4, 10))

    with pytest.raises(
        RuntimeError, match=r"weight should have at least three dimensions"
    ):
        F.conv1d(torch.randn(4, 20), torch.randn(4, 3))

    with pytest.raises(
        RuntimeError,
        match=r"Expected 2D \(unbatched\) or 3D \(batched\) input to conv1d",
    ):
        F.conv1d(torch.randn(20), torch.randn(3))

    """2. S 无需整除 L + 2P - D(K - 1) -1, 但 G 必须整除 C 和 C'"""
