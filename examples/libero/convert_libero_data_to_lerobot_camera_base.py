"""
Convert LIBERO RLDS -> LeRobot, writing BOTH base-frame deltas (`actions`)
and camera-frame deltas (`action_cam`, rotation-only transform).

- Fix: declare `task` in features, or drop it (这里选择声明为 string)
- Fix: force float32 for numeric arrays
- Fix: wrap delta-frame transform with asserts
"""

import shutil
import numpy as np
from lerobot.common.datasets.lerobot_dataset import  LeRobotDataset
import tensorflow_datasets as tfds
import tyro

from pathlib import Path
HF_LEROBOT_HOME = Path("/root/autodl-tmp/my_lerobot_dataset")
REPO_NAME = "lbycdy/libero_camera"

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# ---- 外参（cam -> base），每套子集固定一份 ----
libero_10_T_cam_to_base = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [-0.258174524, 0.0, 0.966098295, 0.0552036108],
     [0.966098295, 0.0, 0.258174524, -2.06578134],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

libero_spatial_T_cam_to_base = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [-0.258174524, 0.0, 0.966098295, -0.120174122],
     [0.966098295, 0.0, 0.258174524, -1.75036630],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

libero_goal_T_cam_to_base = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [-0.258174524, 0.0, 0.966098295, -0.120174122],
     [0.966098295, 0.0, 0.258174524, -1.75036630],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

libero_object_T_cam_to_base = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [-0.258174524, 0.0, 0.966098295, -0.214884654],
     [0.966098295, 0.0, 0.258174524, -1.71357071],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

T_matrices = {
    "libero_10_no_noops": libero_10_T_cam_to_base,
    "libero_spatial_no_noops": libero_spatial_T_cam_to_base,
    "libero_goal_no_noops": libero_goal_T_cam_to_base,
    "libero_object_no_noops": libero_object_T_cam_to_base,
}

# ---- 增量换帧：base -> cam （旋转-only）----
def base_delta_to_cam_delta(action_base7: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    """
    action_base7: [dx,dy,dz, dthx,dthy,dthz, g] (BASE frame delta)
    return: same 7D delta in CAMERA frame
    """
    action_base7 = np.asarray(action_base7, dtype=np.float32)
    assert action_base7.shape[-1] == 7, f"action length must be 7, got {action_base7.shape}"
    R_cb = T_cam_to_base[:3, :3]            # cam -> base
    R_bc = R_cb.T                            # base -> cam
    v_b = action_base7[:3];  w_b = action_base7[3:6]; g = action_base7[6]
    v_c = R_bc @ v_b
    w_c = R_bc @ w_b
    return np.concatenate([v_c, w_c, [g]]).astype(np.float32)

def _safe_decode_lang(x):
    # RLDS 里通常是 bytes；有时已是 str
    try:
        return x.decode("utf-8")
    except Exception:
        return str(x)

def main(data_dir: str, *, push_to_hub: bool = False):
    # 清理旧输出
    output_path = HF_LEROBOT_HOME / REPO_NAME

    if output_path.exists():
        shutil.rmtree(output_path)

    # 定义 schema（新增 `task` 字段为 string，若不需要可删去两处）
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image":       {"dtype": "image",   "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image",   "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state":       {"dtype": "float32", "shape": (8,),          "names": ["state"]},
            "actions":     {"dtype": "float32", "shape": (7,),          "names": ["actions"]},      # base delta
            "action_cam":  {"dtype": "float32", "shape": (7,),          "names": ["action_cam"]},   # cam delta
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_name in RAW_DATASET_NAMES:
        T_cam_to_base = np.asarray(T_matrices[raw_name], dtype=np.float32)
        R = T_cam_to_base[:3, :3]  # 仅用于 sanity check；变换在函数里做

        raw_ds = tfds.load(raw_name, data_dir=data_dir, split="train")
        for episode in raw_ds:
            for step in episode["steps"].as_numpy_iterator():
                # 统一 float32
                action_base = np.asarray(step["action"], dtype=np.float32)      # (7,)
                state       = np.asarray(step["observation"]["state"], dtype=np.float32)  # (8,)

                # base delta -> cam delta
                action_cam = base_delta_to_cam_delta(action_base, T_cam_to_base)

                dataset.add_frame(
                    {
                        "image":        step["observation"]["image"],
                        "wrist_image":  step["observation"]["wrist_image"],
                        "state":        state,
                        "actions":      action_base,
                        "action_cam":   action_cam,
                         "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
