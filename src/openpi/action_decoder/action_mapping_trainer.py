import time
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.openpi.action_decoder.action_mapping_model import ActionMappingMLP
from src.openpi.action_decoder.configs.action_mapping_config import ActionMappingConfig
from src.openpi.action_decoder.local_parquet_action_loader import make_local_action_dataloader
import src.openpi.training.config as _config


class ActionMappingTrainer:
    """动作映射训练器"""

    def __init__(self,
                 train_config: _config.TrainConfig,
                 mapping_config: ActionMappingConfig,
                 device: Optional[str] = None):

        self.train_config = train_config
        self.mapping_config = mapping_config

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 初始化模型
        self.model = ActionMappingMLP(mapping_config).to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=mapping_config.learning_rate,
            weight_decay=mapping_config.weight_decay
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=mapping_config.num_epochs
        )

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        logging.info(f"ActionMappingTrainer 初始化完成")
        logging.info(f"设备: {self.device}")
        logging.info(f"模型参数: {self.model.get_num_parameters():,}")



    def prepare_data(self):
        parquet_root = self.mapping_config.parquet_root


        train_loader = make_local_action_dataloader(
            parquet_root=parquet_root,
            split="train",
            batch_size=self.mapping_config.batch_size,
            max_samples=self.mapping_config.max_samples,
            shuffle_buffer=2048,  # 想要近似shuffle就开这个；不要shuffle可设 0
            num_workers=0,
        )
        val_loader = make_local_action_dataloader(
            parquet_root=parquet_root,
            split="val",
            batch_size=self.mapping_config.batch_size,
            max_samples=(self.mapping_config.max_samples // 5) if self.mapping_config.max_samples else None,
            shuffle_buffer=0,
            num_workers=0,
        )
        return train_loader, val_loader

    def train(self) -> Tuple[List[float], List[float]]:
        """训练模型"""
        train_loader, val_loader = self.prepare_data()
        print("[SANITY] building first batch...", flush=True)
        it = iter(train_loader)
        try:
            x, y = next(it)
        except StopIteration:
            raise RuntimeError("train_loader 是空的：没有产出任何 (action_cam, actions) 样本。")
        print("[SANITY] first train batch shapes:", x.shape, y.shape, flush=True)

        logging.info("开始训练动作映射网络...")
        logging.info("Epoch | Train Loss | Val Loss | Learning Rate | Time")
        logging.info("-" * 65)

        for epoch in range(self.mapping_config.num_epochs):
            start_time = time.time()

            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 验证阶段
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # 更新学习率
            self.scheduler.step()

            # 保存最佳模型
            if val_loss < self.best_val_loss - 1e-5:
                self.best_val_loss = val_loss
                self.save_model('best_action_mapping.pth')

            # 计算时间
            epoch_time = time.time() - start_time

            # 打印进度
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch % 10 == 0 or epoch == self.mapping_config.num_epochs - 1 or epoch < 5:
                logging.info(
                    f"{epoch:5d} | {train_loss:10.6f} | {val_loss:8.6f} | {current_lr:.2e} | {epoch_time:.1f}s")

        # 保存最终模型
        self.save_model('final_action_mapping.pth')

        logging.info("训练完成!")
        return self.train_losses, self.val_losses

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for camera_actions, base_actions in train_loader:
            camera_actions = camera_actions.to(self.device)
            base_actions = base_actions.to(self.device)

            # 前向传播
            pred_base = self.model(camera_actions)
            loss = self.criterion(pred_base, base_actions)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for camera_actions, base_actions in val_loader:
                camera_actions = camera_actions.to(self.device)
                base_actions = base_actions.to(self.device)

                pred_base = self.model(camera_actions)
                loss = self.criterion(pred_base, base_actions)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'mapping_config': self.mapping_config,
            'train_config_name': getattr(self.train_config, "name", None),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        torch.save(checkpoint, filepath)
        logging.info(f"✓ 模型保存到: {filepath} (最佳验证损失: {self.best_val_loss:.6f})")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logging.info(f"✓ 模型从 {filepath} 加载")
        logging.info(f"  最佳验证损失: {self.best_val_loss:.6f}")