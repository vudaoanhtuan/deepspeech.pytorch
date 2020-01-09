import json
import argparse
import warnings
import os.path
import json

import pandas as pd
import torch
from tqdm import tqdm

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

from data.data_loader import SpectrogramParser
from model import DeepSpeech
from decoder import BeamCTCDecoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_file', required=True)
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--lm_path', default=None)
    parser.add_argument('--alpha', type=int, default=0)
    parser.add_argument('--beta', type=int, default=0)
    args = parser.parse_args()
    return args

def transcribe(audio_path, parser, model, decoder, device):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, args.model_file, True)

    parser = SpectrogramParser(model.audio_conf, normalize=True)
    decoder = BeamCTCDecoder(
        model.labels, 
        lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
        beam_width=args.beam_size, num_processes=args.num_worker, blank_index=0
    )

    with open(args.manifest_file) as f:
        data = f.read().split('\n')[:-1]
    
    idx = []
    pred = []
    lbl = []
    for line in tqdm(data):
        vp, tp = line.split(',')
        with open(tp) as f:
            text = f.read().strip()
        decoded_output, decoded_offsets = transcribe(vp, parser, model, decoder, device)
        idx.append(vp)
        pred.append(decoded_output[0][0])
        lbl.append(text)

    df = pd.DataFrame({"id": idx, "predict": pred, "label": lbl})
    df.to_csv(args.manifest_file + '.pred', index=False)