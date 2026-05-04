# local_parquet_action_loader.py
import os, glob, random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.dataset as pads


def _find_parquet_files(root: str):
    root = os.path.expanduser(root)
    files = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    files = sorted([f for f in files if os.path.isfile(f)])
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {root}")
    return files


class ParquetActionPairIterable(IterableDataset):
    """
    直接从本地 parquet 读取 actions / action_cam 两列。
    - 不读取 image/state 等列 => 启动快 + IO 少
    - split: 用简单取模划分 train/val（不依赖 random_split，也不需要全量索引）
    """
    def __init__(
        self,
        parquet_root: str,
        cam_key: str = "actions",
        base_key: str = "actions_base",
        split: str = "train",          # "train" or "val"
        val_mod: int = 10,             # 每 10 条取 1 条做 val
        max_samples: int = 0,
        shuffle_buffer: int = 0,       # >0 开启流式 buffer shuffle
        batch_rows: int = 4096,        # arrow 扫描 batch 大小
        strict_cam: bool = True,
    ):
        assert split in ("train", "val")
        self.parquet_root = os.path.expanduser(parquet_root)
        self.cam_key = cam_key
        self.base_key = base_key
        self.split = split
        self.val_mod = max(2, int(val_mod))
        self.max_samples = max_samples
        self.shuffle_buffer = int(shuffle_buffer)
        self.batch_rows = int(batch_rows)
        self.strict_cam = strict_cam


        files = _find_parquet_files(self.parquet_root)
        print(f"[ParquetLoader] found {len(files)} parquet files under {self.parquet_root}", flush=True)
        self._ds = pads.dataset(files, format="parquet")
        print(f"[ParquetLoader] schema columns: {self._ds.schema.names}", flush=True)

        for k in (self.base_key, self.cam_key):
            if k not in self._ds.schema.names:
                raise KeyError(f"Column '{k}' not in parquet schema. got={self._ds.schema.names}")

    def __iter__(self):
        # 只投影两列（关键：避免把 image/state 读出来）
        cols = [self.base_key, self.cam_key]
        scanner = self._ds.scanner(columns=cols, batch_size=self.batch_rows)

        buf = []
        yielded = 0
        global_i = 0

        for rb in scanner.to_batches():
            # rb: pyarrow.RecordBatch
            base_arr = rb.column(0)
            cam_arr = rb.column(1)

            # 转成 python list（actions/action_cam 通常是 list<float> 或 fixed-size-list）
            # 这里只读两列，开销相对可控
            base_list = base_arr.to_pylist()
            cam_list = cam_arr.to_pylist()

            for b, c in zip(base_list, cam_list):
                # split 划分（不需要随机索引）
                is_val = (global_i % self.val_mod) == 0
                global_i += 1
                if (self.split == "val") != is_val:
                    continue

                if c is None:
                    if self.strict_cam:
                        raise KeyError(f"Missing {self.cam_key} in local parquet sample")
                    c = b

                item = (
                    torch.tensor(np.asarray(c, dtype=np.float32)).view(-1),
                    torch.tensor(np.asarray(b, dtype=np.float32)).view(-1),
                )

                if self.shuffle_buffer > 0:
                    buf.append(item)
                    if len(buf) >= self.shuffle_buffer:
                        random.shuffle(buf)
                        while buf:
                            yield buf.pop()
                            yielded += 1
                            if self.max_samples is not None and yielded >= self.max_samples:
                                return
                else:
                    yield item
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return

        # flush buffer
        if self.shuffle_buffer > 0 and buf:
            random.shuffle(buf)
            for item in buf:
                yield item


def make_local_action_dataloader(
    parquet_root: str,
    split: str,
    batch_size: int,
    max_samples: int = 0,
    shuffle_buffer: int = 0,
    num_workers: int = 0,
):
    ds = ParquetActionPairIterable(
        parquet_root=parquet_root,
        split=split,
        max_samples=max_samples,
        shuffle_buffer=shuffle_buffer,
        strict_cam=True,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
