from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .convlstm import ConvLSTMSequence, group_norm


def _apply_time(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, steps, channels, height, width = x.shape
    y = module(x.reshape(batch * steps, channels, height, width))
    _, out_channels, out_height, out_width = y.shape
    return y.reshape(batch, steps, out_channels, out_height, out_width)


class ConvAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            group_norm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class UpsampleAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "deconv") -> None:
        if mode == "deconv":
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                group_norm(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif mode == "bilinear":
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                group_norm(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")
        super().__init__(*layers)


@dataclass
class StageState:
    state: tuple[torch.Tensor, torch.Tensor]
    spatial_size: tuple[int, int]


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        stage_channels: list[int],
        hidden_channels: list[int],
        kernel_size: int,
        fourier_mode: str,
        alternate_reverse: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(hidden_channels):
            raise ValueError("stage_channels and hidden_channels must have the same length")

        self.alternate_reverse = alternate_reverse
        self.blocks = nn.ModuleList()
        self.cells = nn.ModuleList()

        current_channels = input_channels
        for stage, (stage_out, hidden_out) in enumerate(zip(stage_channels, hidden_channels)):
            stride = 1 if stage == 0 else 2
            self.blocks.append(ConvAct(current_channels, stage_out, stride=stride))
            self.cells.append(
                ConvLSTMSequence(
                    input_channels=stage_out,
                    hidden_channels=hidden_out,
                    kernel_size=kernel_size,
                    fourier_mode=fourier_mode,
                )
            )
            current_channels = hidden_out

    def forward(self, x: torch.Tensor) -> list[StageState]:
        states: list[StageState] = []
        seq = x
        for stage, (block, cell) in enumerate(zip(self.blocks, self.cells)):
            if self.alternate_reverse and stage > 0:
                seq = torch.flip(seq, dims=[1])
            seq = _apply_time(block, seq)
            seq, state = cell(seq)
            states.append(StageState(state=state, spatial_size=seq.shape[-2:]))
        return states


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        hidden_channels: list[int],
        kernel_size: int,
        fourier_mode: str,
        frames: int,
        upsample_mode: str = "deconv",
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        self.frames = frames
        self.output_activation = output_activation

        reversed_hidden = list(reversed(hidden_channels))
        self.cells = nn.ModuleList(
            [
                ConvLSTMSequence(
                    input_channels=channels,
                    hidden_channels=channels,
                    kernel_size=kernel_size,
                    fourier_mode=fourier_mode,
                )
                for channels in reversed_hidden
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpsampleAct(reversed_hidden[i], reversed_hidden[i + 1], mode=upsample_mode)
                for i in range(len(reversed_hidden) - 1)
            ]
        )
        self.head = nn.Conv2d(reversed_hidden[-1], output_channels, kernel_size=1)

    def forward(self, states: list[StageState], output_size: tuple[int, int]) -> torch.Tensor:
        reversed_states = list(reversed(states))
        seq: torch.Tensor | None = None

        for stage, (cell, stage_state) in enumerate(zip(self.cells, reversed_states)):
            h0 = stage_state.state[0]
            seq, _ = cell(
                seq,
                initial_state=stage_state.state,
                steps=self.frames,
                spatial_size=stage_state.spatial_size,
                batch_size=h0.shape[0],
                device=h0.device,
                dtype=h0.dtype,
            )

            if stage < len(self.up_blocks):
                seq = _apply_time(self.up_blocks[stage], seq)
                next_size = reversed_states[stage + 1].spatial_size
                if tuple(seq.shape[-2:]) != tuple(next_size):
                    seq = self._resize_sequence(seq, next_size)
            else:
                seq = _apply_time(self.head, seq)
                if tuple(seq.shape[-2:]) != tuple(output_size):
                    seq = self._resize_sequence(seq, output_size)
                seq = self._activate(seq)

        return seq

    @staticmethod
    def _resize_sequence(seq: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, steps, channels, height, width = seq.shape
        y = F.interpolate(
            seq.reshape(batch * steps, channels, height, width),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return y.reshape(batch, steps, channels, size[0], size[1])

    def _activate(self, seq: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "identity":
            return seq
        if self.output_activation == "relu":
            return F.relu(seq)
        if self.output_activation == "softplus":
            return F.softplus(seq)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(seq)
        raise ValueError(f"Unknown output activation: {self.output_activation}")


class CloudRemovalConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        frames: int = 4,
        stage_channels: list[int] | None = None,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 5,
        fourier_mode: str = "fft_add",
        upsample_mode: str = "deconv",
        alternate_reverse: bool = True,
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        stage_channels = stage_channels or [4, 16, 24]
        hidden_channels = hidden_channels or [16, 24, 24]
        self.frames = frames
        self.encoder = Encoder(
            input_channels=input_channels,
            stage_channels=stage_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            fourier_mode=fourier_mode,
            alternate_reverse=alternate_reverse,
        )
        self.decoder = Decoder(
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            fourier_mode=fourier_mode,
            frames=frames,
            upsample_mode=upsample_mode,
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        states = self.encoder(x)
        return self.decoder(states, output_size=output_size)


def _fourier_mode(model_cfg: dict[str, Any]) -> str:
    mode = model_cfg.get("fourier_mode", "fft_add")
    if mode is True:
        return "fft_add"
    if mode is False or mode is None:
        return "none"
    if mode not in {"none", "fft_add", "fft_concat"}:
        raise ValueError("fourier_mode must be one of: none, fft_add, fft_concat")
    return mode


def build_model(config: dict[str, Any]) -> CloudRemovalConvLSTM:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    return CloudRemovalConvLSTM(
        input_channels=int(model_cfg.get("input_channels", 1)),
        output_channels=int(model_cfg.get("output_channels", 1)),
        frames=int(model_cfg.get("frames", data_cfg.get("frames", 4))),
        stage_channels=list(model_cfg.get("stage_channels", [4, 16, 24])),
        hidden_channels=list(model_cfg.get("hidden_channels", [16, 24, 24])),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        fourier_mode=_fourier_mode(model_cfg),
        upsample_mode=model_cfg.get("upsample_mode", "deconv"),
        alternate_reverse=bool(model_cfg.get("alternate_reverse", True)),
        output_activation=model_cfg.get("output_activation", "identity"),
    )
