import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from seq2seq.src.data_set import DataSets


pad_idx = None
bos_idx = None
eos_idx = None


def generate_batch(data_batch):
    """DataLoader用のコールバック関数
    BOS,EOSの追加、トークン数をPADで埋める
    """
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([bos_idx]), de_item, torch.tensor([eos_idx])], dim=0))
        en_batch.append(torch.cat([torch.tensor([bos_idx]), en_item, torch.tensor([eos_idx])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=pad_idx)
    en_batch = pad_sequence(en_batch, padding_value=pad_idx)
    return de_batch, en_batch


def data_loader(data: DataSets, batch_size, shuffle: bool = True) -> DataLoader:
    """DataLoaderのラッパー"""
    global pad_idx, bos_idx, eos_idx  # pylint: disable=global-statement
    pad_idx = data.pad_idx
    bos_idx = data.bos_idx
    eos_idx = data.eos_idx
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=generate_batch)
