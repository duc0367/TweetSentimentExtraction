import pandas as pd
import os

from config import CFG

pd.set_option('display.max_columns', None)
df = pd.read_csv(os.path.join(CFG.data_folder, 'train.csv'))


def find_position(item):
    text = str(item['text'])
    selected_text = str(item['selected_text'])
    start_id = text.find(selected_text)
    end_id = start_id + len(selected_text)
    item['start_positions'] = start_id
    item['end_positions'] = end_id
    return item


df = df.apply(find_position, axis=1)
df = df.drop(columns=['selected_text'])
df.to_csv(os.path.join(CFG.data_folder, 'train_final.csv'), index=False)
