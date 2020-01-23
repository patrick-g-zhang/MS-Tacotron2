import torch
from model import Tacotron2
from torch.utils.data import DataLoader
from layers import TacotronSTFT, STFT
from hparams import create_hparams
from data_utils import TextMelLoader, TextMelCollate
from train import load_model
from text import text_to_sequence
import numpy as np
hparams = create_hparams()
checkpoint_path = "./checkpoint_99500"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

text = "jan1-wai6-taai3-pei4-gyun6-gwaan1-hai6"
sequence = np.array(text_to_sequence(text, ['cantonese_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
spk_ids = torch.LongTensor([7]).cuda()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, spk_ids)
