"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
import lmdb
import torch.utils.data as ptud


DataLoader = ptud.DataLoader


ChainDataset = ptud.ChainDataset


ConcatDataset = ptud.ConcatDataset


StackDataset = ptud.StackDataset


def lmdb_open_read(
    data_file,
    subdir=False,
    readonly=True,
    readahead=False,
    meminit=False,
    max_spare_txns=4,
    lock=False,
):
    return lmdb.open(
        str(data_file),
        subdir=subdir,
        readonly=readonly,
        readahead=readahead,
        meminit=meminit,
        max_spare_txns=max_spare_txns,
        lock=lock,
    )


def lmdb_open_write(
    dst_file,
    map_size=1024**4,
    subdir=False,
    readonly=False,
    meminit=False,
    lock=True,
):
    return lmdb.open(
        str(dst_file),
        map_size=map_size,
        subdir=subdir,
        readonly=readonly,
        meminit=meminit,
        lock=lock,
    )
