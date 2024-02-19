import torch
import torch.nn as nn

from 04_BTCmodel.modules import MultiHeadAttn, PositionwiseFF, gen_mask, gen_signal


class SelfAttnBlock(nn.Module):
    def __init__(self, timestep, hidden_size, dim_k, dim_v, n_heads, mask,
                 layer_dropout, attn_dropout, act_dropout):
        super(SelfAttnBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttn(hidden_size, dim_k, dim_v, n_heads, mask, attn_dropout)
        self.positionwise_conv = PositionwiseFF(hidden_size, hidden_size, act_dropout)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = nn.LayerNorm([timestep, hidden_size])
        self.layer_norm_ffn = nn.LayerNorm([timestep, hidden_size])

    def forward(self, inputs):
        x_norm = self.layer_norm_mha(inputs)
        y = self.multi_head_attn(x_norm, x_norm, x_norm)

        x = self.dropout(inputs + y)

        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_conv(x_norm)

        outputs = self.dropout(x + y)

        return outputs


class BiDirectionalSelfAttn(nn.Module):
    def __init__(self, timestep, hidden_size, dim_k, dim_v, n_heads,
                 layer_dropout, attn_dropout, act_dropout):
        super(BiDirectionalSelfAttn, self).__init__()
        params = (timestep, hidden_size, dim_k, dim_v, n_heads, gen_mask(timestep),
                  layer_dropout, attn_dropout, act_dropout)
        self.fw_attn_block = SelfAttnBlock(*params)

        params = (timestep, hidden_size, dim_k, dim_v, n_heads,
                  torch.transpose(gen_mask(timestep), dim0=2, dim1=3),
                  layer_dropout, attn_dropout, act_dropout)
        self.bw_attn_block = SelfAttnBlock(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        fw_outputs = self.fw_attn_block(inputs)
        bw_outputs = self.bw_attn_block(inputs)

        x = torch.cat((fw_outputs, bw_outputs), dim=2)
        outputs = self.linear(x)

        return outputs


class BiDirectionalSelfAttnLayers(nn.Module):
    def __init__(self, dim_freq, timestep, hidden_size, dim_k, dim_v, n_heads, n_layers,
                 input_dropout, layer_dropout, attn_dropout, act_dropout):
        super(BiDirectionalSelfAttnLayers, self).__init__()
        params = (timestep, hidden_size, dim_k, dim_v, n_heads,
                  layer_dropout, attn_dropout, act_dropout)

        self.signal = gen_signal(timestep, hidden_size)

        self.linear_emb = nn.Linear(dim_freq, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[BiDirectionalSelfAttn(*params) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm([timestep, hidden_size])
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        x = self.input_dropout(inputs)
        x = self.linear_emb(x)

        x += self.signal[:, :inputs.shape[1], :].type_as(inputs.data)

        x = self.self_attn_layers((x))
        outputs = self.layer_norm(x)

        return outputs


class BTC_model(nn.Module):
    def __init__(self, config):
        super(BTC_model, self).__init__()
        params = (config['timestep'], config['hidden_size'], config['dim_k'], config['dim_v'], config['n_heads'],
                  config['layer_dropout'], config['attn_dropout'], config['act_dropout'])

        self.input_dropout = nn.Dropout(config['input_dropout'])
        self.linear_emb = nn.Linear(config['dim_freq'], config['hidden_size'], bias=False)

        self.signal = gen_signal(config['timestep'], config['hidden_size'])

        self.self_attn_layers = nn.Sequential(*[BiDirectionalSelfAttn(*params) for _ in range(config['n_layers'])])
        self.layer_norm = nn.LayerNorm([config['timestep'], config['hidden_size']])

        self.linear = nn.Linear(config['hidden_size'], config['n_chords'])

    def forward(self, inputs):
        x = self.input_dropout(inputs)
        x = self.linear_emb(x)

        x += self.signal[:, :inputs.shape[1], :].type_as(inputs.data)

        x = self.self_attn_layers(x)
        x = self.layer_norm(x)

        outputs = self.linear(x)

        return outputs
