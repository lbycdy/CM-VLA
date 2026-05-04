# train_action_mapping_main.py
# !/usr/bin/env python3
"""动作映射训练主脚本"""
import argparse
import logging
from src.openpi.training.config import get_config
from src.openpi.action_decoder.action_mapping_trainer_wandb import ActionMappingTrainer
from src.openpi.action_decoder.configs.action_mapping_config import ActionMappingConfig


def main():
    parser = argparse.ArgumentParser(description='训练动作映射网络')
    parser.add_argument('--config', type=str,default='pi0_libero_low_mem_finetune', help='PI配置名称，如 pi0_libero')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, help='设备 (cuda/cpu)')
    # wandb（可选）：记录 train/val loss 曲线
    parser.add_argument('--wandb', action='store_true', help='启用 wandb 记录训练曲线')
    parser.add_argument('--wandb-project', type=str, default='action_mapping', help='wandb 项目名')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity（可选）')
    parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name（可选）')
    parser.add_argument('--wandb-tags', type=str, default=None, help='逗号分隔 tags（可选）')
    parser.add_argument('--wandb-mode', type=str, default=None, choices=['online','offline','disabled'], help='wandb mode（可选）')
    parser.add_argument('--wandb-watch-freq', type=int, default=0, help='>0 时记录梯度/参数的频率(步)')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,  # ✅ 关键：即使 TF/absl 先配置过 logging，也强制覆盖
    )
    print("[BOOT] train_action_mapping.py entered main()")  # ✅ 再加个 print 保底

    # 获取PI官方配置
    train_config = get_config(args.config)
    logging.info(f"使用配置: {args.config}")

    # 创建动作映射配置
    mapping_config = ActionMappingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # 创建训练器并开始训练
    wandb_tags = None
    if args.wandb_tags:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]

    trainer = ActionMappingTrainer(
        train_config=train_config,
        mapping_config=mapping_config,
        device=args.device,
        wandb_enable=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=wandb_tags,
        wandb_mode=args.wandb_mode,
        wandb_watch_freq=args.wandb_watch_freq,
    )

    trainer.train()


if __name__ == "__main__":
    main()
