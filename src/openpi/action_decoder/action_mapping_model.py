# action_mapping_model.py
import torch
import torch.nn as nn
from collections import OrderedDict
import logging

from src.openpi.action_decoder.configs.action_mapping_config import ActionMappingConfig


class ActionMappingMLP(nn.Module):
    """动作映射MLP网络"""

    def __init__(self, config: ActionMappingConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_dim = config.input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append((f'linear_{i}', nn.Linear(prev_dim, hidden_dim)))
            layers.append((f'activation_{i}', config.get_activation()))
            layers.append((f'dropout_{i}', nn.Dropout(config.dropout_rate)))
            prev_dim = hidden_dim

        # 输出层
        layers.append(('output', nn.Linear(prev_dim, config.output_dim)))

        self.network = nn.Sequential(OrderedDict(layers))

        # 初始化权重
        self._init_weights()

        logging.info(f"ActionMappingMLP 初始化完成: "
                     f"输入维度={config.input_dim}, 输出维度={config.output_dim}, "
                     f"隐藏层={config.hidden_dims}")

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, camera_action: torch.Tensor) -> torch.Tensor:
        return self.network(camera_action)

    def get_num_parameters(self) -> int:
        """返回参数数量"""
        return sum(p.numel() for p in self.parameters())