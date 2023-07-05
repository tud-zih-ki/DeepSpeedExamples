#!/usr/bin/env python3

from tbparse import SummaryReader
import pandas as pd
import argparse

RUNTIME_NAME = 'Runtime (sec)'

def parse_tb2(logdir, csv_out):
    reader = SummaryReader(logdir, extra_columns=set(['wall_time']))
    df = reader.scalars

    loss_df = df[df['tag'] == 'Train/Samples/train_loss'].reset_index(drop=True)

    loss_df['wall_clock'] = pd.to_datetime(loss_df.wall_time, unit="s")
    first = loss_df.wall_time.iloc[0]
    loss_df[RUNTIME_NAME] = loss_df.wall_time - first

    loss_df.rename(columns={'step': 'Samples', 'value': 'Loss'}, inplace=True)
    loss_df.drop(columns=['tag', 'wall_time', 'wall_clock'], inplace=True)

    if csv_out:
        loss_df.to_csv(csv_out)

    return loss_df


def find_runtime(df, min_iters, loss_limit):
    for d in df.rolling(min_iters, method="table"):
        if d.shape[0] == min_iters and (d['Loss'] < loss_limit).all():
            return d.iloc[-1][RUNTIME_NAME]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse & print TensorBoard event files')
    parser.add_argument('logdir', help='TensorBoard logdir')
    parser.add_argument('--csv_out', help='csv output file')
    args = parser.parse_args()

    data = parse_tb2(args.logdir, args.csv_out)
    print(data)
    print('='*60)
    target_runtime = find_runtime(data, min_iters=2, loss_limit=8.5)
    if target_runtime:
        print(f"Time to reach target loss: {target_runtime:.2f} s")
    else:
        print("Target loss not reached")
