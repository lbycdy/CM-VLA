
import collections

import dataclasses
import logging
import math
import pathlib
import os

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OnScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from src.openpi.action_decoder.action_mapping_inference import ActionMappingInference


# import pandas as pd  # ← 新增：读取 parquet 作为 GT



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

import numpy as np

# ---- 增量换帧：cam -> base （旋转-only）----
def cam_delta_to_base_delta(actions7: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    """
    actions7: [dx,dy,dz, dthx,dthy,dthz, g] (CAMERA frame delta)
    T_cam_to_base: 4x4 齐次变换矩阵，表示从 camera 到 base 的变换
    return: same 7D delta in BASE frame
    """
    actions7 = np.asarray(actions7, dtype=np.float32)
    assert actions7.shape[-1] == 7, f"action length must be 7, got {actions7.shape}"

    R_cb = T_cam_to_base[:3, :3]  # cam -> base 的旋转矩阵

    v_c = actions7[:3]
    w_c = actions7[3:6]
    g = actions7[6]

    # cam -> base
    v_b = R_cb @ v_c
    w_b = R_cb @ w_c

    return np.concatenate([v_b, w_b, [g]]).astype(np.float32)




@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 1 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_cam"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # === 新增：GT 比较相关参数 ===
    #################################################################################################################
    data_root: str = os.path.expanduser("~/.cache/huggingface/lerobot/lbycdy/libero_camera/data")
    compare_gt: bool = True              # 是否对比标签动作
    gt_frame: str = "base"               # "base" -> 比对 parquet['actions']；"camera" -> 比对 parquet['actions']
    align_skip: int = 0                  # 若需要跳过 GT 前若干步用于对齐，这里设置（默认 0）
    save_eval_csv: str = "action_eval_log.csv"  # 汇总保存路径


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # === 新增：初始化动作映射推理器 ===
    action_mapper = ActionMappingInference(
        model_path="/home/lbycdy/work/camerapi/best_action_mapping.pth",  # 你的训练好的模型路径
        device="cuda"  # 或 "cpu"
    )



    # Start evaluation
    total_episodes, total_successes = 0, 0
    eval_rows = []  # 汇总结果

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)


        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            action_plan = collections.deque()

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            # 新增：准备预测动作缓存（基座系）
            pred_actions_base = []

            logging.info(f"Starting episode {task_episodes+1}...")
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        #action_chunk = client.infer(element)["actions"]
                        action_chunk = client.infer(element)["actions"]

                        print(f"[DEBUG] 原始相机坐标系动作: {action_chunk[0]}")

                        # === 使用神经网络转换动作 ===
                        converted_action_chunk = []
                        for cam_action in action_chunk:
                            # 直接调用推理器进行转换
                            base_action = action_mapper.cam_to_base(cam_action)
                            converted_action_chunk.append(base_action)


                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(converted_action_chunk[: args.replan_steps])

                    actions = action_plan.popleft()
                    action_base = actions



                    # 缓存基座系预测动作
                    pred_actions_base.append(np.asarray(action_base, dtype=np.float32))

                    # Execute action in environment (now in base frame)
                    obs, reward, done, info = env.step(action_base.tolist())
                    env.env.render()

                    if done:
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break



            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")

            task_episodes += 1
            total_episodes += 1
            # 更新累积成功数
            if done:
                if "total_successes" in globals():
                    pass
            # 本脚本原输出统计留存

        # （保留原总体统计打印逻辑）
        # 你可以在此处汇总 task 内成功率









def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OnScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
