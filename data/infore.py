import os
import argparse
import json

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('voice_dir')
parser.add_argument('json_file')
parser.add_argument('--min_duration', type=int, default=0)
parser.add_argument('--max_duration', type=int, default=1000)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)

if __name__=='__main__':
    args = parser.parse_args()

    parent_dir = os.path.dirname(args.json_file)
    text_dir = os.path.join(parent_dir, 'transcribe')
    if not os.path.isdir(text_dir):
        os.mkdir(text_dir)
    text_dir = os.path.abspath(text_dir)
    voice_dir = os.path.abspath(args.voice_dir)

    with open(args.json_file) as f:
        data = f.readlines()
    
    voices = []
    texts = []

    for line in tqdm(data):
        conf = json.loads(line)
        text = conf['text']
        voice = conf['key']
        duration = conf['duration']
        if duration > args.min_duration and duration < args.max_duration:
            vp = os.path.join(voice_dir, voice[37:])
            tp = os.path.join(text_dir, voice[37:-4].replace('/','_')+'.txt')
            with open(tp, 'w') as f:
                f.write(text)
            voices.append(vp)
            texts.append(tp)

    train_voice, test_voice, train_text, test_text = train_test_split(
        voices, texts, 
        test_size=args.test_size,
        random_state=args.random_state
    )

    train_df = pd.DataFrame({'voice':train_voice, 'text':train_text})
    test_df = pd.DataFrame({'voice':test_voice, 'text':test_text})

    train_df.to_csv(os.path.join(parent_dir,'train_manifest.csv'), header=False, index=False)
    test_df.to_csv(os.path.join(parent_dir,'test_manifest.csv'), header=False, index=False)