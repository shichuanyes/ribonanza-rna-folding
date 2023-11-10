import argparse
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='FlipDataset',
        description="Randomly flip sequences and corresponding reactivities in the dataset"
    )

    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--flip_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=283)

    args = parser.parse_args()

    print("Reading input file...")
    df = pd.read_csv(args.input_path)

    n = len(df)
    reactivity_columns = [
        column for column in df.columns
        if not column.startswith('reactivity_error') and column.startswith('reactivity')
    ]

    if 'flip' not in df.columns:
        df['flip'] = pd.Series([False for _ in range(n)])
    rng = np.random.default_rng(seed=args.seed)
    indices = rng.permutation(df.index)
    flip_indices = indices[:math.ceil(n * args.flip_ratio)]

    print("Flipping sequences...")
    df.loc[flip_indices, 'sequence'] = df.loc[flip_indices, 'sequence'].str[::-1]

    if len(reactivity_columns) > 0:
        print("Flipping reactivities...")
        for flip_index in tqdm(flip_indices):
            flip_columns = reactivity_columns[:len(df.loc[flip_index, 'sequence'])]
            df.loc[flip_index, flip_columns] = df.loc[flip_index, flip_columns].to_numpy()[::-1]

    df.loc[flip_indices, 'flip'] = ~ df.loc[flip_indices, 'flip']

    print("Writing to file...")
    df.to_csv(args.output_path, index=False)
