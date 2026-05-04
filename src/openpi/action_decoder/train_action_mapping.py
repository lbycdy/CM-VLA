# train_action_mapping_main.py
# !/usr/bin/env python3
"""动作映射训练主脚本"""
import argparse
import logging
from src.openpi.training.config import get_config
from src.openpi.action_decoder.action_mapping_trainer import ActionMappingTrainer
from src.openpi.action_decoder.configs.action_mapping_config import ActionMappingConfig


def main():
    parser = argparse.ArgumentParser(description='训练动作映射网络')
    parser.add_argument('--config', type=str,default='pi0_libero_low_mem_finetune', help='PI配置名称，如 pi0_libero')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, help='设备 (cuda/cpu)')

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
    trainer = ActionMappingTrainer(
        train_config=train_config,
        mapping_config=mapping_config,
        device=args.device
    )

    trainer.train()


if __name__ == "__main__":
    main()