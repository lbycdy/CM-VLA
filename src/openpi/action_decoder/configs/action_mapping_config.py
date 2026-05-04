# configs/action_mapping_config.py
from dataclasses import dataclass
from typing import List
import torch.nn as nn


@dataclass
class ActionMappingConfig:
    # 网络结构
    input_dim: int = 7  # 相机动作维度
    output_dim: int = 7  # 基座动作维度
    hidden_dims: List[int] = None  # 隐藏层维度

    # 训练参数
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    num_epochs: int = 300
    parquet_root: str = "/home/lbycdy/.cache/huggingface/lerobot/lbycdy/libero_20251219actionV5"

    # 数据参数
    train_ratio: float = 0.95
    max_samples: int = None  # 最大样本数，None表示使用所有数据

    # 激活函数类型
    activation: str = "relu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]

    def get_activation(self):
        """获取激活函数"""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()