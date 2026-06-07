import torch
import onnxruntime as ort
from   model import Hubert, export_onnx, distilhubert_official, load_pretrained


@torch.inference_mode()
def test_export():
    net = Hubert().eval()
    export_onnx(net, '/tmp/hubert.onnx', check=False)
    netOrt = ort.InferenceSession('/tmp/hubert.onnx', providers=['CPUExecutionProvider'])
    x      = torch.randn(2, 1, 48000)
    out0   = net(x) 
    out1,  = netOrt.run(None, {'audio': x.numpy()})
    torch.testing.assert_close(out0, torch.from_numpy(out1))


@torch.inference_mode()
def test_official():
    net0 = distilhubert_official().eval()
    net1 = Hubert().eval()
    load_pretrained(net1, net0)
    audio = torch.randn(1, 48000)
    out0 = net0(audio).last_hidden_state
    out1 = net1(audio[None])
    torch.testing.assert_close(out0,out1)