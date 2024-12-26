import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *


class CounterFactDataset(Dataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / "counterfact_rs.json"
        if not cf_loc.exists():
            print(f"{cf_loc} does not exist. Please download.")
            
        assert cf_loc.exists()
        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
