import collections
import dataclasses
import logging
import math
import pathlib
import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import textwrap
import robosuite.utils.transform_utils as T
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
import os
os.environ["MUJOCO_GL"] = "egl"
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
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

def get_ee_state(env, obs):
    """
    返回:
      ee_state: [x, y, z, ax, ay, az]  (轴角姿态)
      ee_pos:   [x, y, z]
      ee_quat:  [x, y, z, w]  (XYZW)
    """
    # ① 优先从观测里直接拿（robosuite 默认就有）
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        ee_pos = np.asarray(obs["robot0_eef_pos"]).copy()
        ee_quat_xyzw = np.asarray(obs["robot0_eef_quat"]).copy()  # XYZW
    else:
        # ② 若观测中没有，就从 sim 里算（前向运动学）
        inner = getattr(env, "env", env)          # OffScreenRenderEnv 里取内层 env
        sim = inner.sim
        eef_id = inner.robots[0].eef_site_id
        ee_pos = sim.data.site_xpos[eef_id].copy()
        ee_mat = sim.data.site_xmat[eef_id].reshape(3, 3).copy()
        ee_quat_xyzw = T.mat2quat(ee_mat)         # 得到 XYZW

    ee_axis_angle = _quat2axisangle(ee_quat_xyzw.copy())  # 你文件里已有这个函数
    ee_state = np.concatenate([ee_pos, ee_axis_angle], axis=0)
    return ee_state, ee_pos, ee_quat_xyzw
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

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(2)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
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

                    img_to_show = img.copy()


                    # Save preprocessed image for replay video
                    replay_images.append(img_to_show)

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
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    if 'action' in locals():
                        # 格式化 action 数组（保留两位小数）
                        action_text = "[" + ", ".join(f"{a:.4f}" for a in np.array(action).flatten()) + "]"

                        # 使用 textwrap.wrap 自动分行（每行大约 60 字符，可调整）
                        wrapped_lines = textwrap.wrap(action_text, width=20)

                        # 绘制每一行到图像上
                        y0, dy = 20, 15  # 起始位置和行距
                        for i, line in enumerate(wrapped_lines):
                            y = y0 + i * dy
                            cv2.putText(
                                img_to_show,
                                line,
                                (2, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,  # 字体大小
                                (0, 255, 0),  # 绿色
                                1,
                                cv2.LINE_AA
                            )




                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    ee_state, ee_pos, ee_quat = get_ee_state(env, obs)
                    # 想打印/记录：
                    # print("EE pos:", ee_pos, "EE quat(xyzw):", ee_quat, "EE state:", ee_state)

                    action_text1 = "[" + ", ".join(f"{a:.4f}" for a in np.array(ee_pos).flatten()) + "]"

                    # 使用 textwrap.wrap 自动分行（每行大约 60 字符，可调整）
                    wrapped_lines1 = textwrap.wrap(action_text1, width=30)

                    # 绘制每一行到图像上
                    y0, dy = 100, 15  # 起始位置和行距
                    for i, line in enumerate(wrapped_lines1):
                        y = y0 + i * dy
                        cv2.putText(
                            img_to_show,
                            line,
                            (2, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,  # 字体大小
                            (0, 255, 255),  # 绿色
                            1,
                            cv2.LINE_AA
                        )

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            episode_dir = pathlib.Path(args.video_out_path) / f"task_{task_id}_episode_{episode_idx}_{suffix}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx, frame in enumerate(replay_images):
                # 写入图片
                enlarged = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                frame_path = episode_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR))

            logging.info(f"Saved {len(replay_images)} frames to {episode_dir}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
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
