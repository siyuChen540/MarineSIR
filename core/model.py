import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

def mkLayer(block_params: dict) -> nn.Sequential:
    """Builds a nn.Sequential block from a dictionary of layer parameters."""
    layers = []
    for layer_name, params in block_params.items():
        if 'pool' in layer_name:
            k, s, p = params
            layer = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            inC, outC, k, st, pad = params
            layer = nn.ConvTranspose2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        elif 'conv' in layer_name:
            inC, outC, k, st, pad = params
            layer = nn.Conv2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        else:
            raise NotImplementedError(f"Layer type not recognized in: {layer_name}")

        if 'relu' in layer_name:
            layers.append((f'relu_{layer_name}', nn.ReLU(inplace=True)))
        elif 'leaky' in layer_name:
            layers.append((f'leaky_{layer_name}', nn.LeakyReLU(negative_slope=0.2, inplace=True)))
    return nn.Sequential(*[layer for _, layer in layers])

class ConvLSTM_cell(nn.Module):
    """A ConvLSTM cell with optional frequency-domain convolution (fconv)."""
    def __init__(self, shape, channels, kernel_size, features_num, fconv=False, frames_len=10, is_cuda=False):
        super().__init__()
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.is_cuda = is_cuda
        groups_num = max(1, (4 * self.features_num) // 4)
        channel_num = 4 * self.features_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )
        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(2 * (self.channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(8 * self.features_num, 4 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )

    def forward(self, inputs=None, hidden_state=None):
        hx, cx = self._init_hidden(inputs, hidden_state)
        output_frames = []
        for t in range(self.frames_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.channels, self.shape[0], self.shape[1], device=hx.device)
            else:
                x = inputs[t].to(hx.device)
            hy, cy = self._step(x, hx, cx)
            output_frames.append(hy)
            hx, cy = hy, cy
        return torch.stack(output_frames), (hy, cy)

    def _init_hidden(self, inputs, hidden_state):
        if hidden_state is not None: return hidden_state
        bsz = inputs.size(1) if inputs is not None else 1
        device = 'cuda' if self.is_cuda and torch.cuda.is_available() else 'cpu'
        hx = torch.zeros(bsz, self.features_num, self.shape[0], self.shape[1], device=device)
        cx = torch.zeros_like(hx)
        return hx, cx

    def _step(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        concat = torch.cat((x, hx), dim=1)
        gates_out = self.conv(concat)
        if self.fconv:
            fft_dim = (-2, -1)
            freq = torch.fft.rfftn(concat, dim=fft_dim, norm='ortho')
            freq = torch.stack((freq.real, freq.imag), dim=-1)
            freq = freq.permute(0, 1, 4, 2, 3).contiguous()
            N, C, _, H, W2 = freq.size()
            freq = freq.view(N, -1, H, W2)
            ffc_conv = self.semi_conv(freq)
            ifft_shape = ffc_conv.shape[-2:]
            ffc_out = torch.fft.irfftn(torch.complex(ffc_conv, torch.zeros_like(ffc_conv)), s=ifft_shape, dim=fft_dim, norm='ortho')
            ffc_out_resize = F.interpolate(ffc_out, size=gates_out.size()[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat((ffc_out_resize, gates_out), 1)
            gates_out = self.global_conv(combined)
        
        in_gate, forget_gate, hat_cell_gate, out_gate = torch.split(gates_out, self.features_num, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        hat_cell_gate = torch.tanh(hat_cell_gate)
        out_gate = torch.sigmoid(out_gate)
        cy = (forget_gate * cx) + (in_gate * hat_cell_gate)
        hy = out_gate * torch.tanh(cy)
        return hy, cy

class Encoder(nn.Module):
    """Encoder composed of multiple Conv/Pool layers and ConvLSTM cells."""
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.transpose(0, 1)
        hidden_states = []
        for i in range(self.block_num):
            if i > 0: inputs = torch.flip(inputs, [0])
            fnum, bsz, ch, h, w = inputs.size()
            reshaped = inputs.view(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            processed = processed.view(fnum, bsz, nch, nh, nw)
            outputs, state_stage = self.convlstm_cells[i](processed, None)
            hidden_states.append(state_stage)
            inputs = outputs
        return tuple(hidden_states)

class Decoder(nn.Module):
    """Decoder composed of ConvLSTM cells and UpConv (DeConv) layers."""
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, hidden_states):
        hidden_states = hidden_states[::-1]
        inputs = None
        for i in range(self.block_num):
            outputs, _ = self.convlstm_cells[i](inputs, hidden_states[i])
            seq_num, bsz, ch, h, w = outputs.size()
            reshaped = outputs.view(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            inputs = processed.view(seq_num, bsz, nch, nh, nw)
        return inputs.transpose(0, 1)

class ED(nn.Module):
    """Encoder-Decoder architecture for sequence-to-sequence prediction."""
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        state_list = self.encoder(inputs)
        output = self.decoder(state_list)
        return output
