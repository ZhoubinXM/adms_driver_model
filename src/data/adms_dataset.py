from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np
from scipy.special import erf, erfinv


class ADMSDataset(Dataset):
    """ADMS general dataset"""

    def __init__(self, data_path: str, label_process: str = 'log', verbose: bool = True, **kwargs) -> None:
        """Dataset Init func

        Args:
            data_path (str): dataset csv save path
        """
        super().__init__(**kwargs)
        self.mode = data_path.split('/')[-1].split('.')[-2]
        if verbose:
            print(f"[{self.__doc__}][{self.mode}]: Load data from: {data_path}")
        self.dataset = pd.read_pickle(data_path)
        self.data_path = data_path
        self.label_process = label_process
        # self.process_label()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        data = self.dataset.iloc[index]
        agents = np.array(data.agents).astype(np.float32)
        lanes = np.array(data.lanes).astype(np.float32)
        dense_fea = data.loc['c_DANADSts':'c_HOSts'].values.astype(int)
        sparse_fea = data.loc['l_age':'l_YawRateSAERps'].values.astype(np.float32)
        tot = data.tot_cat_equal_width
        tob = data.tot_cat_equal_width
        return data.uuid, agents, lanes, dense_fea, sparse_fea, tot, tob

    def process_label(self):
        if self.label_process == 'quantile':
            tot_qt = QuantileTransformer()
            tob_qt = QuantileTransformer()
            tot_data = self.dataset.tot
            tob_data = self.dataset.tob
            self.dataset.tot = tot_qt.fit(tot_data).transform(tot_data)
            self.datseet.tob = tob_qt.fit(tob_data).transform(tob_data)


class QuantileTransformer:

    def __init__(self):
        self.sorted_x = None

    def fit(self, x):
        # 对输入数据进行排序，并保存
        self.sorted_x = torch.sort(x)[0]
        return self

    def transform(self, x):
        # 计算每个数据点的分位数，并转换为标准正态分布的对应值
        quantiles = np.searchsorted(self.sorted_x, x) / len(self.sorted_x)
        return np.sqrt(2) * erfinv(2 * quantiles - 1)

    def inverse_transform(self, x):
        # 计算每个数据点在标准正态分布中的分位数，并转换回原始数据的对应值
        quantiles = (torch.erf(x / np.sqrt(2)) + 1) / 2
        return self.sorted_x[(quantiles * (len(self.sorted_x) - 1)).long()]


if __name__ == "__main__":
    adms_dataset = ADMSDataset("../data/test.pkl")
    # 创建一个偏态分布的数据集
    data = np.random.exponential(size=1000)
    # 创建分位数转换器
    qt = QuantileTransformer()
    # 对数据进行转换
    data_trans = qt.fit(data).transform(data)
    # 对数据进行逆转换
    data_original = qt.inverse_transform(data_trans)
