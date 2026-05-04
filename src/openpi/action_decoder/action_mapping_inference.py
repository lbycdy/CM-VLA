# action_mapping_inference.py
import torch
from src.openpi.action_decoder.action_mapping_model import ActionMappingMLP
import numpy as np

class ActionMappingInference:
    """动作映射推理器"""

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ 动作映射模型加载完成，使用设备: {self.device}")

    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        mapping_config = checkpoint['mapping_config']

        model = ActionMappingMLP(mapping_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def cam_to_base(self, cam_action):
        """
        将相机坐标系动作转换为基座坐标系动作

        Args:
            cam_action: 相机坐标系动作，可以是 list, numpy array 或 tensor
                       shape: (7,) 或 (batch_size, 7)

        Returns:
            base_action: 基座坐标系动作，numpy array
        """
        # 转换为 tensor
        if isinstance(cam_action, (list, np.ndarray)):
            cam_tensor = torch.tensor(cam_action, dtype=torch.float32)
        else:
            cam_tensor = cam_action

        # 确保正确的维度
        if cam_tensor.ndim == 1:
            cam_tensor = cam_tensor.unsqueeze(0)

        # 转移到设备并推理
        cam_tensor = cam_tensor.to(self.device)

        with torch.no_grad():
            base_tensor = self.model(cam_tensor)

        # 转换回 numpy
        base_action = base_tensor.cpu().numpy()

        # 如果输入是单个动作，输出也应该是单个动作
        if base_action.shape[0] == 1 and len(cam_action) == 7:
            base_action = base_action[0]

        return base_action