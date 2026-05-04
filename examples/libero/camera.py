import os
os.environ["MUJOCO_GL"] = "osmesa"
import dataclasses
import logging
import math
import pathlib
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OnScreenRenderEnv
import numpy as np
import tyro

# 相机内参函数（你给出的）
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

LIBERO_ENV_RESOLUTION = 256  # resolution used to render data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )

    seed: int = 7  # Random Seed (for reproducibility)

    # 相机参数（如果你想改相机名或分辨率，可以从命令行/这里修改）
    camera_name: str = "agentview"
    camera_height: int = LIBERO_ENV_RESOLUTION
    camera_width: int = LIBERO_ENV_RESOLUTION


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OnScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    保留原来的工具函数（如果你后面还要用，可以留着；不用也可以删）
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_env_and_compute_intrinsics(args: Args):
    """
    只做一件事：
      1. 创建一个 LIBERO 仿真环境
      2. 计算并打印指定相机的内参矩阵
    """
    # 固定随机种子
    np.random.seed(args.seed)

    # 1. 初始化一个 task_suite，并取第一个 task（随便哪个任务，只是为了把场景和相机关起来）
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(0)

    # 2. 建环境（只建一次）
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Using task: {task_description}")

    # 3. 拿到底层 mujoco sim
    #    在你原来的代码里是用 env.env.sim 来访问
    sim = env.env.sim

    # 4. 计算相机内参
    camera_name = args.camera_name
    H = args.camera_height
    W = args.camera_width

    K = get_camera_intrinsic_matrix(sim, camera_name, H, W)

    print("\n================ 相机内参矩阵 =================")
    print(f"Camera name     : {camera_name}")
    print(f"Image size (HxW): {H} x {W}")
    print("K (3x3):")
    print(K)
    print("================================================\n")

    # 如果你想后续再用 env，可以在这里 return env, K
    return env, K


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    build_env_and_compute_intrinsics(args)
