from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class ADMSDataset(Dataset):
    """ADMS general dataset"""

    def __init__(self,
                 data_path: str,
                 label_process: str = 'log',
                 verbose: bool = True,
                 **kwargs) -> None:
        """Dataset Init func

        Args:
            data_path (str): dataset csv save path
        """
        super().__init__(**kwargs)
        self.mode = data_path.split('/')[-1].split('.')[-2]
        if verbose:
            print(
                f"[{self.__doc__}][{self.mode}]: Load data from: {data_path}")
        self.dataset = pd.read_pickle(data_path)
        self.data_path = data_path
        self.label_process = label_process

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        data = self.dataset.iloc[index]
        agents = np.array(data.agents).astype(np.float32)
        lanes = np.array(data.lanes).astype(np.float32)
        dense_fea = data.iloc[4:16].values.astype(int)
        sparse_fea = data.iloc[16:-2].values.astype(np.float32)
        tot = data.iloc[-2]
        tob = data.iloc[-1]
        return data.uuid, agents, lanes, dense_fea, sparse_fea, tot, tob


if __name__ == "__main__":
    adms_dataset = ADMSDataset("../data/test.pkl")
