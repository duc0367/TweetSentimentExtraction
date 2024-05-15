import pandas as pd
import os

from config import CFG

df = pd.read_csv(os.path.join(CFG.data_folder, 'train.csv'))
max_length = df['text'].str.len().max()
print(max_length)
