import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNAPredictDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='RNAPredict',
        description="Predict RNA sequence reactivity"
    )

    parser.add_argument('data_path')
    parser.add_argument('model_path', nargs='?', default='model.pt')
    parser.add_argument('save_path', nargs='?', default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--flip_ratio', type=float, default=0.5)

    args = parser.parse_args()

    assert 0.0 <= args.flip_ratio <= 1.0

    print("Reading dataset...")
    df = pd.read_csv(args.data_path)

    dataloader = DataLoader(RNAPredictDataset(df, flip_ratio=args.flip_ratio), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = torch.load(args.model_path).to(device)

    predictions = np.empty(shape=(df['id_max'].max() + 1, 2))

    model.eval()
    with torch.inference_mode():
        curr = 0
        for sequences, flips in tqdm(dataloader):
            sequences = sequences.to(device)

            mask = sequences.sum(dim=-1) == 0

            outputs = model(sequences, mask)

            outputs = outputs.cpu().numpy()
            mask = mask.cpu().numpy()

            outputs[flips] = outputs[flips][:, ::-1]
            mask[flips] = mask[flips][:, ::-1]

            outputs = outputs[~ mask]
            predictions[curr:curr + outputs.shape[0], :] = outputs

            curr += outputs.shape[0]

    df = pd.DataFrame({
        'id': np.arange(df['id_max'].max() + 1),
        'reactivity_DMS_MaP': predictions[:, 0],
        'reactivity_2A3_MaP': predictions[:, 1]
    })
    df.to_csv(args.save_path, index=False)
