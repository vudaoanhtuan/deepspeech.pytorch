import os 
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    df = pd.read_csv(args.csv_file)
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)

    train_fp = "train_" + args.csv_file
    test_fp = "test_" + args.csv_file

    train_df.to_csv(train_fp, index=False, header=False)
    test_df.to_csv(test_fp, index=False, header=False)
