# 参考官方数据加载方式
import libero
from libero import get_libero_path
from libero.data import get_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from collections import OrderedDict
import libero
from libero.data import get_dataset
from libero import get_libero_path


class ActionMappingMLP(nn.Module):
    """简单的MLP网络，学习从相机坐标到基坐标的动作映射"""

    def __init__(self,
                 input_dim=7,  # 默认7维，包括位置和旋转
                 hidden_dims=[512, 256, 128],
                 output_dim=7,
                 dropout_rate=0.1):
        super(ActionMappingMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f'linear_{i}', nn.Linear(prev_dim, hidden_dim)))
            layers.append((f'relu_{i}', nn.ReLU()))
            layers.append((f'dropout_{i}', nn.Dropout(dropout_rate)))
            prev_dim = hidden_dim

        # 输出层
        layers.append(('output', nn.Linear(prev_dim, output_dim)))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, camera_action):
        return self.network(camera_action)


class LiberoActionDataset(Dataset):
    """基于OpenPI官方代码的数据集加载"""

    def __init__(self,
                 task_name=None,
                 data_path=None,
                 normalize=True,
                 use_camera_obs=True):
        super(LiberoActionDataset, self).__init__()

        self.normalize = normalize
        self.use_camera_obs = use_camera_obs

        # 使用官方方式加载数据集
        self.dataset = self._load_libero_dataset(task_name, data_path)
        self.camera_actions, self.base_actions = self._extract_actions()

        # 数据统计信息
        if self.normalize:
            self.camera_mean = torch.mean(self.camera_actions, dim=0)
            self.camera_std = torch.std(self.camera_actions, dim=0)
            self.base_mean = torch.mean(self.base_actions, dim=0)
            self.base_std = torch.std(self.base_actions, dim=0)

            self.camera_actions = self._normalize(self.camera_actions, self.camera_mean, self.camera_std)
            self.base_actions = self._normalize(self.base_actions, self.base_mean, self.base_std)

        print(f"数据集加载完成: {len(self.camera_actions)} 个样本")
        print(f"相机动作形状: {self.camera_actions.shape}")
        print(f"基座动作形状: {self.base_actions.shape}")

    def _load_libero_dataset(self, task_name, data_path):
        """使用官方方法加载LIERO数据集"""
        try:
            # 方法1: 使用任务名加载
            if task_name is not None:
                print(f"正在加载任务: {task_name}")
                dataset = get_dataset(
                    task_name=task_name,
                    use_camera_obs=self.use_camera_obs,
                )
            # 方法2: 使用数据路径加载
            elif data_path is not None:
                print(f"正在从路径加载: {data_path}")
                dataset = get_dataset(
                    dataset_path=data_path,
                    use_camera_obs=self.use_camera_obs,
                )
            else:
                raise ValueError("必须提供task_name或data_path")

            return dataset

        except Exception as e:
            print(f"官方数据集加载失败: {e}")
            print("尝试直接加载数据文件...")
            return self._load_direct_data(data_path)

    def _load_direct_data(self, data_path):
        """直接加载数据文件的后备方法"""
        # 这里可以根据你的实际数据格式实现
        # 假设你的数据已经处理成numpy数组或torch tensor
        raise NotImplementedError("请根据你的数据格式实现此方法")

    def _extract_actions(self):
        """从数据集中提取相机动作和基座动作"""
        camera_actions_list = []
        base_actions_list = []

        # 遍历数据集中的所有episode
        for i in range(len(self.dataset)):
            try:
                # 获取一个episode的数据
                episode = self.dataset[i]

                # 根据官方数据结构提取动作
                # 这里需要根据你的实际数据结构调整
                if hasattr(episode, 'actions'):
                    base_actions = episode.actions  # 基座标动作
                elif 'actions' in episode:
                    base_actions = episode['actions']
                else:
                    continue

                # 提取相机坐标动作
                if hasattr(episode, 'camera_actions'):
                    camera_actions = episode.camera_actions
                elif 'camera_actions' in episode:
                    camera_actions = episode['camera_actions']
                elif 'action_cam' in episode:
                    camera_actions = episode['action_cam']
                else:
                    # 如果没有相机动作，可能需要从观测中计算
                    camera_actions = self._compute_camera_actions(episode)

                if camera_actions is not None and base_actions is not None:
                    camera_actions_list.append(torch.from_numpy(camera_actions).float())
                    base_actions_list.append(torch.from_numpy(base_actions).float())

            except Exception as e:
                print(f"处理episode {i} 时出错: {e}")
                continue

        if len(camera_actions_list) == 0:
            raise ValueError("未能提取到有效的动作数据")

        camera_actions = torch.cat(camera_actions_list, dim=0)
        base_actions = torch.cat(base_actions_list, dim=0)

        return camera_actions, base_actions

    def _compute_camera_actions(self, episode):
        """如果没有直接的相机动作，从观测中计算"""
        # 这里需要根据你的具体需求实现
        # 例如：从相机观测和基座观测之间的关系计算
        return None

    def _normalize(self, data, mean, std):
        """标准化数据"""
        return (data - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.camera_actions)

    def __getitem__(self, idx):
        return self.camera_actions[idx], self.base_actions[idx]


class ActionMappingTrainer:
    """基于官方风格的训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 初始化模型
        self.model = ActionMappingMLP(
            input_dim=config.get('input_dim', 7),
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            output_dim=config.get('output_dim', 7)
        ).to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )

        self.best_val_loss = float('inf')

        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader=None, num_epochs=100):
        """训练循环"""

        train_losses = []
        val_losses = []

        print("开始训练动作映射网络...")
        print("Epoch | Train Loss | Val Loss | Learning Rate")
        print("-" * 50)

        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                val_losses.append(val_loss)

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model('best_action_mapping.pth')
            else:
                val_loss = 0.0

            # 更新学习率
            self.scheduler.step()

            # 打印进度
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch % 10 == 0 or epoch == num_epochs - 1 or epoch < 5:
                print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:8.6f} | {current_lr:.2e}")

        print("训练完成!")
        return train_losses, val_losses

    def _train_epoch(self, train_loader):
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

        return total_loss / num_batches

    def _validate_epoch(self, val_loader):
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

        return total_loss / num_batches

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"✓ 模型保存到: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"✓ 模型从 {filepath} 加载")


def main():
    """主训练函数"""
    # 配置参数 - 参考官方配置风格
    config = {
        # 数据配置
        'task_name': 'libero_10',  # 或你的具体任务名
        'data_path': None,  # 如果使用路径而不是任务名

        # 模型配置
        'input_dim': 7,
        'output_dim': 7,
        'hidden_dims': [512, 256, 128],

        # 训练配置
        'batch_size': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 300,
        'train_ratio': 0.9,

        # 设备配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 加载数据集
    print("正在使用OpenPI官方方式加载数据集...")
    dataset = LiberoActionDataset(
        task_name=config['task_name'],
        data_path=config['data_path'],
        normalize=True
    )

    # 分割数据集
    dataset_size = len(dataset)
    train_size = int(config['train_ratio'] * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"数据集统计:")
    print(f"  总样本: {dataset_size}")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  批次大小: {config['batch_size']}")

    # 初始化训练器
    trainer = ActionMappingTrainer(config)

    # 开始训练
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs']
    )

    # 保存最终模型
    trainer.save_model('final_action_mapping.pth')

    # 输出训练结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")
    print(f"最佳验证损失: {min(val_losses):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()