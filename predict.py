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

    args = parser.parse_args()

    print("Reading dataset...")
    df = pd.read_csv(args.data_path)

    dataloader = DataLoader(RNAPredictDataset(df), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = torch.load(args.model_path).to(device)

    predictions = np.empty(shape=(0, 2))

    for inputs, seq_lengths in tqdm(dataloader):
        inputs, seq_lengths = inputs.to(device), seq_lengths.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            for i in range(outputs.size(0)):
                predictions = np.append(predictions, outputs[i, :seq_lengths[i]].cpu().numpy(), axis=0)

    df = pd.DataFrame({
        'id': np.arange(predictions.shape[0]),
        'reactivity_DMS_MaP': predictions[:, 1],
        'reactivity_2A3_MaP': predictions[:, 0]
    })
    df.to_csv(args.save_path, index=False)
