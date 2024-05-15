from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd

from config import CFG


class TSEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        start_positions = item['start_positions']
        end_positions = item['end_positions']
        encoded_text = self.tokenizer(item['text'], truncation=True, max_length=CFG.max_length,
                                      return_offsets_mapping=True)
        offset = encoded_text['offset_mapping']

        start_idx = 0
        end_idx = 0

        for idx, (start, end) in enumerate(offset):
            if start <= start_positions < end:
                start_idx = idx
            if start < end_positions <= end:
                end_idx = idx

        return encoded_text, start_idx, end_idx
