# 参考 OpenPI 官方（LeRobotDataset）数据加载方式
# 你已经把数据放到 HuggingFace 官方缓存路径/或 $HF_LEROBOT_HOME 下时，
# 这里直接改 repo_id 就能加载对应数据集。

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# LeRobot 的 import 在不同版本里路径不一样，这里做兼容（对应 openpi issue #617 的修复思路）
try:
    from lerobot.constants import HF_LEROBOT_HOME  # 新版
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:  # pragma: no cover
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # 旧版（部分版本）
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        # 更旧的版本可能只有 LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME as HF_LEROBOT_HOME  # type: ignore
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


class ActionMappingMLP(nn.Module):
    """简单的MLP网络，学习从相机坐标到基坐标的动作映射"""

    def __init__(
        self,
        input_dim=7,  # 默认7维，包括位置和旋转
        hidden_dims=[512, 256, 128],
        output_dim=7,
        dropout_rate=0.1,
    ):
        super(ActionMappingMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f"linear_{i}", nn.Linear(prev_dim, hidden_dim)))
            layers.append((f"relu_{i}", nn.ReLU()))
            layers.append((f"dropout_{i}", nn.Dropout(dropout_rate)))
            prev_dim = hidden_dim

        # 输出层
        layers.append(("output", nn.Linear(prev_dim, output_dim)))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, camera_action):
        return self.network(camera_action)


class LiberoActionDataset(Dataset):
    """基于 OpenPI 官方训练同款方式：用 LeRobotDataset 从 HuggingFace 缓存加载数据。

    说明：
    - OpenPI 的 LIBERO 训练数据是 LeRobotDataset 格式（非 libero.get_dataset 那套）
    - 真正的“本地路径”由 $HF_LEROBOT_HOME 控制；你也可以在这里传 hf_lerobot_home 覆盖
    - 这个脚本默认从样本里取：
        base_actions:  优先 actions_base / base_actions / actions
        camera_actions: 优先 actions / camera_actions / actions_camera / action_cam
      如果找不到 camera_actions，会退化成 camera_actions = base_actions（仅用于先跑通流程）
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        hf_lerobot_home: str | None = None,
        task_filter: str | list[str] | None = None,
        normalize: bool = True,
        max_samples: int | None = None,
    ):
        super(LiberoActionDataset, self).__init__()

        self.normalize = normalize
        self.repo_id = repo_id
        self.task_filter = task_filter

        # 可选：覆盖 HF_LEROBOT_HOME（你已经把数据放到官方路径的话，这里可以不传）
        if hf_lerobot_home is not None:
            os.environ["HF_LEROBOT_HOME"] = hf_lerobot_home

        # 加载 LeRobotDataset（优先本地缓存；如果你没联网，建议先确保缓存齐全）
        self.dataset = self._load_lerobot_dataset(repo_id)

        # 先构建索引（可选按 task 过滤）
        self.indices = self._build_indices(max_samples=max_samples)

        # 抽取动作对
        self.camera_actions, self.base_actions = self._extract_actions()

        # 标准化
        if self.normalize:
            self.camera_mean = torch.mean(self.camera_actions, dim=0)
            self.camera_std = torch.std(self.camera_actions, dim=0)
            self.base_mean = torch.mean(self.base_actions, dim=0)
            self.base_std = torch.std(self.base_actions, dim=0)

            self.camera_actions = self._normalize(self.camera_actions, self.camera_mean, self.camera_std)
            self.base_actions = self._normalize(self.base_actions, self.base_mean, self.base_std)

        print(f"数据集加载完成: {len(self.camera_actions)} 个样本")
        print(f"repo_id: {self.repo_id}")
        print(f"HF_LEROBOT_HOME: {os.environ.get('HF_LEROBOT_HOME', str(HF_LEROBOT_HOME))}")
        print(f"相机动作形状: {tuple(self.camera_actions.shape)}")
        print(f"基座动作形状: {tuple(self.base_actions.shape)}")

    def _load_lerobot_dataset(self, repo_id: str):
        """用 LeRobotDataset 加载（兼容不同版本的参数）。"""
        # 有些版本支持 local_files_only=True，有些不支持；这里做一次容错
        try:
            return LeRobotDataset(repo_id, local_files_only=True)  # type: ignore[arg-type]
        except TypeError:
            return LeRobotDataset(repo_id)
        except Exception as e:
            raise RuntimeError(
                f"LeRobotDataset 加载失败: repo_id={repo_id}.\n"
                f"请检查：1) 你本地缓存是否完整 2) HF_LEROBOT_HOME 是否正确 3) repo_id 是否写对。\n"
                f"原始错误: {e}"
            ) from e

    def _build_indices(self, max_samples: int | None = None) -> list[int]:
        """构建样本索引，可按 task_filter 过滤。"""
        all_indices = list(range(len(self.dataset)))

        if self.task_filter is None:
            indices = all_indices
        else:
            if isinstance(self.task_filter, str):
                filters = [self.task_filter]
            else:
                filters = list(self.task_filter)

            indices = []
            for i in all_indices:
                sample = self.dataset[i]
                task = sample.get("task", None)
                if task is None:
                    continue
                # 允许“完全匹配”或“包含子串”
                if any((task == f) or (f in task) for f in filters):
                    indices.append(i)

            if len(indices) == 0:
                raise ValueError(
                    f"task_filter={self.task_filter} 没有过滤到任何样本。\n"
                    "提示：OpenPI 的 libero 数据里 task 通常是自然语言指令字符串，而不是 'libero_10' 这种 suite 名。"
                )

        if max_samples is not None:
            indices = indices[:max_samples]

        return indices

    @staticmethod
    def _as_float_tensor(x) -> torch.Tensor:
        """把 numpy / list / torch 都转成 float32 tensor。"""
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32, device="cpu")
        # 兼容 numpy / list
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    def _extract_actions(self):
        """从 LeRobotDataset 中提取 camera_actions 和 base_actions（按 frame）。"""
        camera_actions_list = []
        base_actions_list = []

        warned_fallback = False

        for idx in self.indices:
            sample = self.dataset[idx]

            # base actions：优先找更“像基座”的字段名
            base = None
            for k in ("actions_base", "base_actions", "actions"):
                if k in sample:
                    base = sample[k]
                    break

            if base is None:
                continue

            # camera actions：优先找更“像相机坐标”的字段名
            cam = None
            for k in ("actions", "camera_actions", "actions_camera", "action_cam"):
                if k in sample:
                    cam = sample[k]
                    break

            if cam is None:
                # 没有的话先跑通流程：退化为 cam == base
                cam = base
                if not warned_fallback:
                    print(
                        "[警告] 数据里没有找到 camera_actions/actions_camera 等字段，\n"
                        "       将暂时使用 camera_actions = base_actions 作为占位（只为跑通流程）。\n"
                        "       如果你确实要做 camera->base 映射，请把 camera_actions 的计算方式补上。"
                    )
                    warned_fallback = True

            cam_t = self._as_float_tensor(cam)
            base_t = self._as_float_tensor(base)

            # 确保是 1D (action_dim,)
            if cam_t.ndim != 1:
                cam_t = cam_t.view(-1)
            if base_t.ndim != 1:
                base_t = base_t.view(-1)

            camera_actions_list.append(cam_t)
            base_actions_list.append(base_t)

        if len(camera_actions_list) == 0:
            raise ValueError("未能提取到有效的动作数据（actions/actions_base 等字段缺失或为空）")

        camera_actions = torch.stack(camera_actions_list, dim=0)
        base_actions = torch.stack(base_actions_list, dim=0)
        return camera_actions, base_actions

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
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 初始化模型
        self.model = ActionMappingMLP(
            input_dim=config.get("input_dim", 7),
            hidden_dims=config.get("hidden_dims", [512, 256, 128]),
            output_dim=config.get("output_dim", 7),
        ).to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
        )

        self.best_val_loss = float("inf")

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
                    self.save_model("best_action_mapping.pth")
            else:
                val_loss = 0.0

            # 更新学习率
            self.scheduler.step()

            # 打印进度
            current_lr = self.optimizer.param_groups[0]["lr"]
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

        return total_loss / max(1, num_batches)

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

        return total_loss / max(1, num_batches)

    def save_model(self, filepath):
        """保存模型"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            filepath,
        )
        print(f"✓ 模型保存到: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"✓ 模型从 {filepath} 加载")


def main():
    """主训练函数"""
    # 配置参数
    config = {
        # 数据配置：只要改 repo_id 就行
        "repo_id": "lbycdy/libero_20251219actionV4",  # 你也可以改成你自己的 repo_id（或你推到 hub 的路径）
        "hf_lerobot_home": None,  # 可选：例如 "/data/hf_cache/lerobot"；不填则用系统默认/你环境变量
        "task_filter": None,  # 可选：按自然语言 task 过滤（不是 'libero_10' 这种）
        "max_samples": None,  # 可选：调试时先取少量样本

        # 模型配置（如果 action 维度不是 7，会在加载后自动覆盖）
        "input_dim": 7,
        "output_dim": 7,
        "hidden_dims": [512, 256, 128],

        # 训练配置
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 300,
        "train_ratio": 0.9,

        # 设备配置
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # 加载数据集
    print("正在使用 OpenPI 官方（LeRobotDataset）方式加载数据集...")
    dataset = LiberoActionDataset(
        repo_id=config["repo_id"],
        hf_lerobot_home=config["hf_lerobot_home"],
        task_filter=config["task_filter"],
        normalize=True,
        max_samples=config["max_samples"],
    )

    # 自动对齐动作维度（避免你手填 7 但数据不是 7 维导致报错）
    config["input_dim"] = int(dataset.camera_actions.shape[-1])
    config["output_dim"] = int(dataset.base_actions.shape[-1])

    # 分割数据集
    dataset_size = len(dataset)
    train_size = int(config["train_ratio"] * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("数据集统计:")
    print(f"  总样本: {dataset_size}")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  action_dim: in={config['input_dim']} out={config['output_dim']}")

    # 初始化训练器
    trainer = ActionMappingTrainer(config)

    # 开始训练
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
    )

    # 保存最终模型
    trainer.save_model("final_action_mapping.pth")

    # 输出训练结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")
    print(f"最佳验证损失: {min(val_losses):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()