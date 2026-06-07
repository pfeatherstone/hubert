import pytest
import torch
import onnxruntime as ort
from   model import Hubert, export_onnx


@torch.inference_mode()
def test_export():
    net = Hubert().eval()
    export_onnx(net, '/tmp/hubert.onnx', check=False)
    netOrt = ort.InferenceSession('/tmp/hubert.onnx', providers=['CPUExecutionProvider'])
    x      = torch.randn(2, 1, 48000)
    out0   = net(x) 
    out1,  = netOrt.run(None, {'audio': x.numpy()})
    torch.testing.assert_close(out0, torch.from_numpy(out1))