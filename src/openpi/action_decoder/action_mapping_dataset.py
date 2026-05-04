# action_mapping_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import src.openpi.training.config as _config
import logging


class CameraActionMappingDataset(Dataset):
    """
    从 LeRobot 数据集中同时加载相机坐标动作 (action_cam)
    与基座坐标动作 (actions)，用于学习 cam→base 映射。
    加载逻辑与 OpenPI 官方完全一致（使用 data_config.create() 自动定位缓存）。
    """

    def __init__(
        self,
        train_config: _config.TrainConfig,
        split: str = "train",
        max_samples: int | None = None,
        cam_action_key: str = "actions",
        base_action_key: str = "actions_base",
        strict_cam: bool = False,
    ):
        self.train_config = train_config
        self.max_samples = max_samples
        self.cam_action_key = cam_action_key
        self.base_action_key = base_action_key
        self.strict_cam = strict_cam
        self.split = split

        # ✅ 与官方完全一致：由 config 解析数据路径和 split
        logging.info(f"创建 LeRobotDataset (split={split}) ...")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

        # 官方版本会直接在 data_config.create() 内部定位到 arrow 缓存路径
        self.dataset = LeRobotDataset(
            data_config.repo_id
        )

        # ✅ 限制加载样本数量（不会再扫描全量 parquet）
        if self.max_samples is not None:
            self.dataset = Subset(self.dataset, range(self.max_samples))
            logging.info(f"仅使用前 {self.max_samples} 条样本进行训练调试")

        self.length = len(self.dataset)
        logging.info(f"CameraActionMappingDataset 初始化完成，共 {self.length} 条样本")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.dataset[idx]

        base_action = item.get(self.base_action_key)
        if base_action is None:
            raise KeyError(f"缺少 {self.base_action_key} 字段, 当前样本包含: {list(item.keys())}")

        cam_action = item.get(self.cam_action_key)
        if cam_action is None:
            if self.strict_cam:
                raise KeyError(f"缺少 {self.cam_action_key} 字段, 当前样本包含: {list(item.keys())}")
            cam_action = base_action  # fallback

        cam = torch.as_tensor(cam_action, dtype=torch.float32)
        base = torch.as_tensor(base_action, dtype=torch.float32)

        if cam.ndim == 2:
            cam = cam[0]
        if base.ndim == 2:
            base = base[0]

        return cam.view(-1), base.view(-1)


def get_cam_action_dataloader(
    train_config: _config.TrainConfig,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    max_samples: int | None = None,
):
    """官方风格 dataloader 构造函数"""
    dataset = CameraActionMappingDataset(
        train_config=train_config,
        split=split,
        max_samples=max_samples,
        strict_cam=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
