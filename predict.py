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
    print()

    dataloader = DataLoader(RNAPredictDataset(df), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = torch.load(args.model_path).to(device)

    result = []
    for inputs in tqdm(dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.flatten(outputs, end_dim=-2)
            result.append(outputs)
    predictions = torch.cat(result, dim=0).cpu().numpy()

    df = pd.DataFrame({
        'id': np.arange(predictions.shape[0]),
        'reactivity_DMS_MaP': predictions[:, 1],
        'reactivity_2A3_MaP': predictions[:, 0]
    })
    df.to_csv(args.save_path, index=False)
