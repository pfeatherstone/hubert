import  torch
from    torchcodec.decoders import AudioDecoder
from    model import Hubert, distilhubert_official, load_pretrained

net0 = distilhubert_official().eval()
net1 = Hubert().eval()
load_pretrained(net1, net0)
audio = AudioDecoder('/home/pf/Downloads/speech_orig.wav', sample_rate=16000, num_channels=1).get_all_samples().data
out0 = net0(audio).last_hidden_state
out1 = net1(audio[None])
torch.testing.assert_close(out0,out1)