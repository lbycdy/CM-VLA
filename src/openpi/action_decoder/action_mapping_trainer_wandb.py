import time
import logging
from typing import List, Tuple, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.openpi.action_decoder.action_mapping_model import ActionMappingMLP
from src.openpi.action_decoder.configs.action_mapping_config import ActionMappingConfig
from src.openpi.action_decoder.local_parquet_action_loader import make_local_action_dataloader
import src.openpi.training.config as _config


class ActionMappingTrainer:
    """动作映射训练器（支持可选 wandb 记录 loss 曲线）"""

    def __init__(self,
                 train_config: _config.TrainConfig,
                 mapping_config: ActionMappingConfig,
                 device: Optional[str] = None,
                 wandb_enable: bool = False,
                 wandb_project: str = "action_mapping",
                 wandb_entity: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_tags: Optional[List[str]] = None,
                 wandb_mode: Optional[str] = None,
                 wandb_watch_freq: int = 0):

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
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # wandb（可选）
        self._wandb = None
        self.wandb_run = None
        self.global_step = 0

        if wandb_enable:
            try:
                import wandb  # type: ignore
                self._wandb = wandb

                cfg = asdict(self.mapping_config)
                cfg.update({
                    "train_config_name": getattr(self.train_config, "name", None),
                    "device": str(self.device),
                })

                init_kwargs = {
                    "project": wandb_project,
                    "config": cfg,
                }
                if wandb_entity:
                    init_kwargs["entity"] = wandb_entity
                if wandb_name:
                    init_kwargs["name"] = wandb_name
                if wandb_tags:
                    init_kwargs["tags"] = wandb_tags
                if wandb_mode:
                    init_kwargs["mode"] = wandb_mode  # online/offline/disabled

                self.wandb_run = wandb.init(**init_kwargs)

                # 需要的话可以记录梯度（会增加开销）
                if wandb_watch_freq and wandb_watch_freq > 0:
                    wandb.watch(self.model, log="gradients", log_freq=wandb_watch_freq)

                logging.info("✓ 已启用 wandb 记录训练指标 (train/val loss 曲线)")
            except ImportError:
                logging.warning("wandb 未安装：请先 `pip install wandb`，或关闭 --wandb")
            except Exception as e:
                logging.warning(f"wandb 初始化失败，将继续训练但不记录 wandb：{e}")
                self._wandb = None
                self.wandb_run = None

        logging.info("ActionMappingTrainer 初始化完成")
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

        try:
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
                    if self.wandb_run is not None:
                        self._wandb_log_model('best_action_mapping.pth', alias='best')

                # 计算时间
                epoch_time = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]['lr']

                # wandb 记录（每个 epoch 记录一次，wandb 会自动生成折线图）
                if self.wandb_run is not None and self._wandb is not None:
                    self._wandb.log(
                        {
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "lr": current_lr,
                            "epoch_time_sec": epoch_time,
                        },
                        step=epoch,
                    )

                if epoch % 10 == 0 or epoch == self.mapping_config.num_epochs - 1 or epoch < 5:
                    logging.info(
                        f"{epoch:5d} | {train_loss:10.6f} | {val_loss:8.6f} | {current_lr:.2e} | {epoch_time:.1f}s")

            # 保存最终模型
            self.save_model('final_action_mapping.pth')
            if self.wandb_run is not None:
                self._wandb_log_model('final_action_mapping.pth', alias='final')

            logging.info("训练完成!")
            return self.train_losses, self.val_losses
        finally:
            if self.wandb_run is not None and self._wandb is not None:
                try:
                    self._wandb.finish()
                except Exception:
                    pass

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
            self.global_step += 1

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

    def _wandb_log_model(self, filepath: str, alias: str):
        """把模型 checkpoint 作为 artifact 记录到 wandb（可选）"""
        if self.wandb_run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(
                "action_mapping_model",
                type="model",
                metadata={"alias": alias, "best_val_loss": float(self.best_val_loss)},
            )
            artifact.add_file(filepath)
            self.wandb_run.log_artifact(artifact, aliases=[alias])
        except Exception as e:
            logging.warning(f"wandb artifact 记录失败: {e}")

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