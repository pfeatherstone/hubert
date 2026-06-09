from   copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.export import Dim
from   torch.nn.utils.parametrize import remove_parametrizations
from   einops import rearrange
import onnxruntime as ort


##########################################################################################################
##########################################################################################################
### MODEL
##########################################################################################################
##########################################################################################################


def count_parameters(net: nn.Module, trainableOnly: bool = False):
    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainableOnly)


def repeat(m, count):
    return nn.Sequential(*[deepcopy(m) for _ in range(count)])


def Conv(c1, c2, k, s, p=0,norm="none"):
    return nn.Sequential(nn.Conv1d(c1,c2,k,s,padding=p,bias=False),
                         nn.GroupNorm(c2,c2) if norm=="group" else nn.Identity(),
                         nn.GELU())


class PosEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos    = nn.Conv1d(768,768,kernel_size=128,padding=64,groups=16)
        self.norm   = nn.LayerNorm(768)
        self.drop   = nn.Dropout(0.1)
    def forward(self, x):
        x = x + F.gelu(self.pos(x))[:,:,:-1]
        x = self.norm(x.transpose(1,2))
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads, drop):
        super().__init__()
        self.dim_head   = dim // num_heads
        self.H          = num_heads
        self.drop       = drop
        self.to_qkv     = nn.Linear(dim, 3*self.H*self.dim_head, bias=True)
        self.to_out     = nn.Linear(self.H*self.dim_head, dim, bias=True)

    def forward(self, x):
        drop    = self.drop if self.training else 0.0
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.H), self.to_qkv(x).chunk(3,-1))
        x       = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=False)
        x       = self.to_out(rearrange(x, 'b h t d -> b t (h d)'))
        x       = F.dropout(x, p=drop, training=self.training)
        return x


def FeedForward(dim, ff_mult, drop):
    return nn.Sequential(nn.Linear(dim, int(dim*ff_mult)),
                         nn.GELU(),
                         nn.Dropout(drop),
                         nn.Linear(int(dim*ff_mult), dim),
                         nn.Dropout(drop))
    

class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ff_mult, drop):
        super().__init__()
        self.attn   = Attention(dim, num_heads, drop)
        self.ff     = FeedForward(dim, ff_mult, drop)
        self.norms  = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
    def forward(self, x):
        x = self.norms[0](x + self.attn(x))
        x = self.norms[1](x + self.ff(x))
        return x
    

def Hubert():
    return nn.Sequential(Conv(1,512,k=10,s=5,norm="group"),
                         repeat(Conv(512,512,k=3,s=2), 4),
                         repeat(Conv(512,512,k=2,s=2), 2),
                         nn.Conv1d(512,768,1),
                         PosEmbedding(),
                         repeat(EncoderLayer(768, 12, 4, 0.1), 2))


##########################################################################################################
##########################################################################################################
### EXPORT
##########################################################################################################
##########################################################################################################


@torch.inference_mode()
def export_onnx(net: nn.Module, onnx_path: str, check:bool = True):
    x = torch.randn(4, 1, 16000)
    _ = net(x) # compile einops if necessary

    prog = torch.export.export(net, (x,), dynamic_shapes=((Dim.DYNAMIC, Dim.STATIC, Dim.DYNAMIC),))
    torch.onnx.export(prog, opset_version=23, input_names=['audio'], output_names=['feats']).save(onnx_path)

    if check:
        netOrt = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        x      = torch.randn(2, 1, 48000)
        out0   = net(x) 
        out1,  = netOrt.run(None, {'audio': x.numpy()})
        torch.testing.assert_close(out0, torch.from_numpy(out1), atol=1e-4, rtol=1e-2)


@torch.no_grad()
def save_raw(net, file:str):
    with open(file, 'wb') as f:
        for p in net.parameters():
            if p.dim()==3: p.transpose(1,2).contiguous().numpy().tofile(f)
            else: p.numpy().tofile(f)


@torch.no_grad()
def save_cpp(net, file:str, name:str, values_per_line:int = 8):
    data = []
    for p in net.parameters():
        if p.dim()==3: data.append(p.transpose(1,2).contiguous().flatten().numpy())
        else: data.append(p.flatten().numpy())
    data = np.concatenate(data)
    with open(file, 'wt') as f:
        f.write("#include <cstddef>\n\n")
        f.write(f"alignas(32) extern const float {name}_WEIGHTS[] = {{\n")
        for i in range(0, len(data), values_per_line):
            row = data[i:i+values_per_line]
            literals = ", ".join(f"{np.float32(v).item():.9g}f" for v in row)
            f.write(f"        {literals},\n")

        f.write("};\n\n")
        f.write(f"extern const std::size_t {name}_SIZE = sizeof({name}_WEIGHTS) / sizeof(float);\n\n")
        

##########################################################################################################
##########################################################################################################
### OFFICIAL
##########################################################################################################
##########################################################################################################


def remove_all_parametrizations(module: torch.nn.Module):
    def remove_all_parametrizations_(m: torch.nn.Module):
        for child in m.children():
            if hasattr(child, "parametrizations"):
                for param_name in list(child.parametrizations.keys()):
                    remove_parametrizations(child, param_name)
            remove_all_parametrizations_(child)
    module_new = deepcopy(module)
    remove_all_parametrizations_(module_new)
    return module_new


def distilhubert_official():
    from transformers import AutoModel
    net = AutoModel.from_pretrained("ntu-spml/distilhubert").eval()
    net = remove_all_parametrizations(net)
    return net


def copy_params_(n1, n2):
    for p1, p2 in zip(n1.parameters(), n2.parameters(), strict=True):
        p1.data.copy_(p2.data)


def copy_wb_(c1, c2):
    c1.weight.data.copy_(c2.weight.data)
    c1.bias.data.copy_(c2.bias.data)


def copy_attn_(a1, a2):
    wq, wk, wv = a2.q_proj.weight.data, a2.k_proj.weight.data, a2.v_proj.weight.data
    bq, bk, bv = a2.q_proj.bias.data, a2.k_proj.bias.data, a2.v_proj.bias.data
    a1.to_qkv.weight.data.copy_(torch.cat((wq,wk,wv),0))
    a1.to_qkv.bias.data.copy_(torch.cat((bq,bk,bv),0))
    copy_wb_(a1.to_out, a2.out_proj)


def copy_encoder_(e1, e2):
    copy_attn_(  e1.attn,     e2.attention)
    copy_params_(e1.ff,       e2.feed_forward)
    copy_params_(e1.norms[0], e2.layer_norm)
    copy_params_(e1.norms[1], e2.final_layer_norm)


def load_pretrained(net0, net1):
    copy_params_(net0[0:3], net1.feature_extractor)
    net0[3].weight.data.copy_(net1.feature_projection.projection.weight.data.unsqueeze(-1))
    net0[3].bias.data.copy_(net1.feature_projection.projection.bias.data)
    copy_wb_(net0[4].pos,  net1.encoder.pos_conv_embed.conv)
    copy_wb_(net0[4].norm, net1.encoder.layer_norm)
    copy_encoder_(net0[5][0], net1.encoder.layers[0])
    copy_encoder_(net0[5][1], net1.encoder.layers[1])