"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro


# ========= 6D pose <-> 4x4 齐次矩阵 =========
# 约定：action 的 6D 位姿为 [x, y, z, roll, pitch, yaw]，欧拉角顺序 xyz(Rx,Ry,Rz)

def euler_xyz_to_R(roll, pitch, yaw):
    cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    # R = Rz * Ry * Rx （只要前后一致即可）
    return Rz @ Ry @ Rx


def R_to_euler_xyz(R):
    # 简单 xyz 提取（假设不在奇异点附近）
    sy = -R[2, 0]
    cy = np.sqrt(max(0.0, 1.0 - sy**2))
    pitch = np.arctan2(sy, cy)

    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw  = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def pose6_to_T(pose6):
    x, y, z, roll, pitch, yaw = pose6
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_xyz_to_R(roll, pitch, yaw)
    T[:3, 3]  = np.array([x, y, z], dtype=np.float32)
    return T


def T_to_pose6(T):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = R_to_euler_xyz(T[:3, :3])
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)


REPO_NAME = "lbycdy/libero_camera_pose"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

libero_10_T_cam_to_base =[[ 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
 [-2.58174524e-01, -1.43315650e-17, 9.66098295e-01, 5.52036108e-02],
 [ 9.66098295e-01, 5.36292286e-17, 2.58174524e-01, -2.06578134e+00],
 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

libero_spatial_T_cam_to_base =[[ 0.00000000e+00,1.00000000e+00,0.00000000e+00,0.00000000e+00],
 [-2.58174524e-01,-1.43315650e-17,9.66098295e-01,-1.20174122e-01],
 [ 9.66098295e-01,5.36292286e-17,2.58174524e-01,-1.75036630e+00],
 [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]


libero_goal_T_cam_to_base =[[ 0.00000000e+00,1.00000000e+00,0.00000000e+00,0.00000000e+00],
 [-2.58174524e-01,-1.43315650e-17,9.66098295e-01, -1.20174122e-01],
 [ 9.66098295e-01,5.36292286e-17,2.58174524e-01, -1.75036630e+00],
 [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]


libero_object_T_cam_to_base =[[ 0.00000000e+00,1.00000000e+00,0.00000000e+00,0.00000000e+00],
 [-2.58174524e-01,-1.43315650e-17,9.66098295e-01,-2.14884654e-01],
 [ 9.66098295e-01,5.36292286e-17,2.58174524e-01,-1.71357071e+00],
 [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]

T_matrices = {
    "libero_10_no_noops": libero_10_T_cam_to_base,
    "libero_spatial_no_noops": libero_spatial_T_cam_to_base,
    "libero_goal_no_noops": libero_goal_T_cam_to_base,
    "libero_object_no_noops": libero_object_T_cam_to_base,
}

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    # print(output_path)
    # exit()
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "state_cam": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state_cam"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        # --- [2] 获取手眼标定矩阵 ---
        T_cam_to_base = np.array(T_matrices[raw_dataset_name], dtype=np.float32)
        T_base_to_cam = np.linalg.inv(T_cam_to_base)

        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():

                state_base = step["observation"]["state"].astype(np.float32)

                # 基座坐标系下的 6D 位姿 + gripper
                pose_base = state_base[:6]  # [x, y, z, roll, pitch, yaw]
                gripper1 = state_base[6]
                gripper2 = state_base[7]

                # 1) 6D -> 4x4 齐次矩阵（base 下末端位姿）
                T_base_ee = pose6_to_T(pose_base)

                # 2) 用手眼标定矩阵把位姿从 base 变到 camera
                #    T_cam_ee = T_base_to_cam * T_base_ee
                T_cam_ee = T_base_to_cam @ T_base_ee

                # 3) 再从 4x4 提回 6D pose
                pose_cam = T_to_pose6(T_cam_ee)

                state_cam = np.concatenate([pose_cam, [gripper1], [gripper2]]).astype(np.float32)

                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "state_cam": state_cam,
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
