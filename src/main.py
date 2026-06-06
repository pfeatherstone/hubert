from   copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn.utils.parametrize import remove_parametrizations
from   torch.export import Dim
from   torchcodec.decoders import AudioDecoder
from   einops import rearrange
from   transformers import AutoModel
import onnxruntime as ort


class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'


def count_parameters(net: nn.Module, trainableOnly: bool = False):
    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainableOnly)


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


def apply(x, layers):
    for l in layers:
        x = l(x)
    return x


def repeat(m, count):
    return nn.Sequential(*[deepcopy(m) for _ in range(count)])


def Conv(c1, c2, k, s, p=0,g=1,norm="none"):
    return nn.Sequential(nn.Conv1d(c1,c2,k,s,padding=p,groups=g,bias=False),
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
    

class Hubert(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0  = Conv(1,512,k=10,s=5,norm="group")
        self.b1  = repeat(Conv(512,512,k=3,s=2), 4)
        self.b2  = repeat(Conv(512,512,k=2,s=2), 2)
        self.b3  = nn.Conv1d(512,768,1)
        self.pos = PosEmbedding()
        self.enc = nn.Sequential(*[EncoderLayer(768, 12, 4, 0.1) for _ in range(2)])

    def forward(self, x):
        x = apply(x, [self.b0,self.b1,self.b2,self.b3])
        x = self.pos(x)
        x = self.enc(x)
        return x


def copy_params(n1, n2):
    for p1, p2 in zip(n1.parameters(), n2.parameters(), strict=True):
        p1.data.copy_(p2.data)


def copy_wb(c1, c2):
    c1.weight.data.copy_(c2.weight.data)
    c1.bias.data.copy_(c2.bias.data)


def copy_attn(a1, a2):
    wq, wk, wv = a2.q_proj.weight.data, a2.k_proj.weight.data, a2.v_proj.weight.data
    bq, bk, bv = a2.q_proj.bias.data, a2.k_proj.bias.data, a2.v_proj.bias.data
    a1.to_qkv.weight.data.copy_(torch.cat((wq,wk,wv),0))
    a1.to_qkv.bias.data.copy_(torch.cat((bq,bk,bv),0))
    copy_wb(a1.to_out, a2.out_proj)


def copy_encoder(e1, e2):
    copy_attn(  e1.attn,     e2.attention)
    copy_params(e1.ff,       e2.feed_forward)
    copy_params(e1.norms[0], e2.layer_norm)
    copy_params(e1.norms[1], e2.final_layer_norm)


def load_params(net0, net1):
    copy_params(nn.ModuleList([net0.b0, net0.b1, net0.b2]), net1.feature_extractor)
    net0.b3.weight.data.copy_(net1.feature_projection.projection.weight.data.unsqueeze(-1))
    net0.b3.bias.data.copy_(net1.feature_projection.projection.bias.data)
    copy_wb(net0.pos.pos, net1.encoder.pos_conv_embed.conv)
    copy_params(net0.pos.norm, net1.encoder.layer_norm)
    copy_encoder(net0.enc[0], net1.encoder.layers[0])
    copy_encoder(net0.enc[1], net1.encoder.layers[1])


@torch.inference_mode()
def export_onnx(net: nn.Module, onnx_path: str, check:bool = True):
    x = torch.randn(4, 1, 16000)
    _ = net(x) # compile einops if necessary

    print(bcolors.OKGREEN, f"Exporting {type(net).__name__} ...", bcolors.ENDC)
    prog = torch.export.export(net, (x,), dynamic_shapes={'x' : (Dim.DYNAMIC, Dim.STATIC, Dim.DYNAMIC)})
    torch.onnx.export(prog, opset_version=23, input_names=['audio'], output_names=['feats']).save(onnx_path)
    print(bcolors.OKGREEN, f"Exporting {type(net).__name__} ... Done", bcolors.ENDC)

    if check:
        print(bcolors.OKGREEN, f"Checking {type(net).__name__} ...", bcolors.ENDC)
        netOrt = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        x      = torch.randn(2, 1, 48000)
        out0   = net(x) 
        out1,  = netOrt.run(None, {'audio': x.numpy()})
        torch.testing.assert_close(out0, torch.from_numpy(out1), atol=1e-4, rtol=1e-2)
        print(bcolors.OKGREEN, "Checking with onnxruntime... Done", bcolors.ENDC)


name = "ntu-spml/distilhubert"
net0 = remove_all_parametrizations(AutoModel.from_pretrained(name)).eval() 
net1 = Hubert().eval()
print(f"net0 size {count_parameters(net0)}")
print(f"net1 size {count_parameters(net1)}")
load_params(net1, net0)
export_onnx(net1, '/tmp/hubert.onnx', check=True)

audio = AudioDecoder('/home/pf/Downloads/speech_orig.wav', sample_rate=16000, num_channels=1).get_all_samples().data
with torch.inference_mode():
    out0 = net0(audio).last_hidden_state
    out1 = net1(audio[None])
torch.testing.assert_close(out0,out1)