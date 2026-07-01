from __future__ import annotations

import torch
from torch import nn


def group_norm(channels: int, group_size: int = 4) -> nn.GroupNorm:
    groups = max(1, channels // group_size)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SpectralGateConv(nn.Module):
    """Frequency-domain convolution used to enrich ConvLSTM gates.

    The original project converted real/imaginary FFT parts into channels,
    convolved them, then transformed them back. This module keeps that idea but
    reconstructs directly to the original spatial size, avoiding interpolation
    artifacts and reducing temporary tensors.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size, padding=padding),
            group_norm(out_channels * 2),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        freq = torch.view_as_real(freq)
        freq = freq.permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, height, width // 2 + 1)

        freq = self.net(freq)
        freq = freq.reshape(x.shape[0], self.out_channels, 2, height, width // 2 + 1)
        freq = freq.permute(0, 1, 3, 4, 2).contiguous()
        freq = torch.view_as_complex(freq)
        return torch.fft.irfft2(freq, s=(height, width), dim=(-2, -1), norm="ortho")


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        fourier_mode: str = "none",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.fourier_mode = fourier_mode

        padding = (kernel_size - 1) // 2
        gate_channels = 4 * hidden_channels
        combined_channels = input_channels + hidden_channels

        self.local_gate = nn.Sequential(
            nn.Conv2d(combined_channels, gate_channels, kernel_size, padding=padding),
            group_norm(gate_channels),
        )

        if fourier_mode in {"fft_add", "fft_concat"}:
            self.spectral_gate = SpectralGateConv(combined_channels, gate_channels, kernel_size=1)
        else:
            self.spectral_gate = None

        if fourier_mode == "fft_concat":
            self.fuse_gate = nn.Sequential(
                nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1),
                group_norm(gate_channels),
            )
        else:
            self.fuse_gate = None

    def forward_step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h, c = self.init_state(x.shape[0], x.shape[-2:], x.device, x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.local_gate(combined)

        if self.spectral_gate is not None:
            spectral = self.spectral_gate(combined)
            if self.fourier_mode == "fft_add":
                gates = gates + spectral
            elif self.fourier_mode == "fft_concat":
                gates = self.fuse_gate(torch.cat([gates, spectral], dim=1))

        i, f, g, o = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        shape = (batch_size, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


class ConvLSTMSequence(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        fourier_mode: str = "none",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            fourier_mode=fourier_mode,
        )

    def forward(
        self,
        inputs: torch.Tensor | None,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        steps: int | None = None,
        spatial_size: tuple[int, int] | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if inputs is not None:
            batch_size, steps, _, height, width = inputs.shape
            device = inputs.device
            dtype = inputs.dtype
            spatial_size = (height, width)
        else:
            if initial_state is not None:
                batch_size = initial_state[0].shape[0]
                spatial_size = initial_state[0].shape[-2:]
                device = initial_state[0].device
                dtype = initial_state[0].dtype
            if steps is None or batch_size is None or spatial_size is None or device is None or dtype is None:
                raise ValueError("Decoder ConvLSTM needs steps, batch_size, spatial_size, device and dtype")

        state = initial_state
        outputs: list[torch.Tensor] = []
        zero_x: torch.Tensor | None = None
        for step in range(int(steps)):
            if inputs is None:
                if zero_x is None:
                    zero_x = torch.zeros(
                        int(batch_size),
                        self.input_channels,
                        int(spatial_size[0]),
                        int(spatial_size[1]),
                        device=device,
                        dtype=dtype,
                    )
                x = zero_x
            else:
                x = inputs[:, step]
            state = self.cell.forward_step(x, state)
            outputs.append(state[0])

        return torch.stack(outputs, dim=1), state

