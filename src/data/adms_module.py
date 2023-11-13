import lightning.pytorch as pl
from typing import Optional
from torch.utils.data import DataLoader
from src.data.adms_dataset import ADMSDataset
import os


class ADMSDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 **kwargs) -> None:
        super(ADMSDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.stage = kwargs['stage']

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.stage == 'debug':
            self.train_dataset = ADMSDataset(
            os.path.join(self.root, 'mini.pkl'))
            self.val_dataset = ADMSDataset(os.path.join(self.root, 'mini.pkl'))
            self.test_dataset = ADMSDataset(os.path.join(self.root, 'mini.pkl'))
        else:
            self.train_dataset = ADMSDataset(
                os.path.join(self.root, 'train_val.pkl'))
            self.val_dataset = ADMSDataset(os.path.join(self.root, 'test.pkl'))
            self.test_dataset = ADMSDataset(os.path.join(self.root, 'test.pkl'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          drop_last=False)
