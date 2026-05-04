from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset
from action_mapping_dataset import CameraActionMappingDataset

ds = LeRobotDataset("lbycdy/libero_20251219actionV4")
subset = Subset(ds, range(100))
for i in range(5):
    item = subset[i]
    print(item.keys())
    print(item["actions"].shape, item.get("actions_base", None))
