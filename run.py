import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import os.path as pt

parser = argparse.ArgumentParser()
parser.add_argument("-f","--file", help="a path to a file with a dataset",
                    type=str,default=pt.join('data','test_100000.csv'))
parser.add_argument("-r", "--random", help="take N random molecules from the dataset",
                    type=int,default=1000)
parser.add_argument("-b", "--beam-size", help="Beam size for Transformer",
                    type=int,default=5)

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_file = args.file
data = pd.read_csv(data_file, sep=",")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

M = torch.load(pt.join("models","iupac2smiles_model.pt"), map_location=device)
M.device = device

def count_acc(df, beam=1):
    count = 0
    for iupac, smiles in tqdm(zip(df['target'], df['input']),total=len(df)):
        try:
            smiles_pred, prob = M.predict_single(iupac, beam=beam)
            if smiles in smiles_pred:
                count += 1
        except:
            pass
    accuracy = str(round(count/df.shape[0]*100, 2)) + '%'
    print(f'Accuracy is {accuracy} on the {str(df.shape[0])} random examples')

count_acc(data.sample(n=args.random),beam=args.beam_size)
