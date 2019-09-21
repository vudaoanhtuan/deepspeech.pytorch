import os 
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('voice_dir')
    parser.add_argument('text_file')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    text_dir = os.path.dirname(args.text_file)
    transcribe_dir = os.path.join(text_dir, 'transcribe')
    if not os.path.isdir(transcribe_dir):
        os.mkdir(transcribe_dir)
    transcribe_dir = os.path.abspath(transcribe_dir)
    voice_dir = os.path.abspath(args.voice_dir)

    with open(args.text_file) as f:
        data = f.read().split('\n')[:-1]

    voice_path = []
    text_path = []

    for line in data:
        v = line[:15]
        t = line[16:]
        vp = os.path.join(voice_dir, v+'.wav')
        tp = os.path.join(transcribe_dir, v+'.txt')
        with open(tp, 'w') as f:
            f.write(t)
        voice_path += [vp]
        text_path += [tp]

    df = pd.DataFrame({'voice': voice_path, 'text': text_path})
    df.to_csv(args.text_file+'.csv', index=False, header=False)
        