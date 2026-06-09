import  time
import  numpy as np
import  torch
from    torchcodec.decoders import AudioDecoder
import  onnxruntime as ort
from    model import count_parameters, Hubert, distilhubert_official, load_pretrained, export_onnx, save_cpp, save_raw


def bench(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    net = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
    
    data = np.random.randn(1,1,24000).astype(np.float32)
    ntests = 100
    s0 = time.time()
    for _ in range(ntests):
        _ = net.run(None, {'audio': data})
    s1 = time.time()
    
    print(f"Bench model {data.shape[-1]*ntests/(s1-s0)} Hz")
    print("Done")


if __name__ == '__main__':
    net0 = distilhubert_official().eval()
    net1 = Hubert().eval()
    print(f"net0 {count_parameters(net0)} params")
    print(f"net1 {count_parameters(net1)} params")
    load_pretrained(net1, net0)
    audio = AudioDecoder('/home/pf/Downloads/speech_orig.wav', sample_rate=16000, num_channels=1).get_all_samples().data
    out0 = net0(audio).last_hidden_state
    out1 = net1(audio[None])
    torch.testing.assert_close(out0,out1)
    save_raw(net1, "hubert.dat")
    export_onnx(net1, '/tmp/hubert.onnx', check=True)
    bench('/tmp/hubert.onnx')