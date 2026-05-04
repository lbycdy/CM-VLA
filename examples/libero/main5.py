import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, OnScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import os
os.environ["MUJOCO_GL"] = "egl"
# 新增：用于在图像上绘字
from PIL import Image, ImageDraw, ImageFont

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


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
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    # 仍然沿用这个参数名，但现在会把“视频目录”当成“图片输出的根目录”
    video_out_path: str = "data/libero/videos"

    seed: int = 7  # Random Seed (for reproducibility)


def _format_array(arr, precision=3):
    arr = np.asarray(arr).astype(float).tolist()
    return "[" + ", ".join(f"{x:+.{precision}f}" for x in arr) + "]"


def _overlay_text_on_image(img_np: np.ndarray, action, eef_pos) -> np.ndarray:
    """
    在图像左上角叠加两行文本：
      1) action（红色）
      2) robot0_eef_pos（青色）
    并绘制半透明底与合适的留白/行距，返回叠加后的 uint8 np.ndarray。
    """
    # 转为 PIL
    pil_img = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(pil_img, mode="RGBA")

    # 字体（优先等宽；缺失则回退默认）
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # 文本内容
    text_action = f"a: {_format_array(action, precision=4)}"
    text_eef = f"eef_pos: {_format_array(eef_pos, precision=4)}"

    # 颜色与布局
    color_action = (255, 90, 90, 255)      # 红
    color_eef = (80, 200, 255, 255)        # 青
    margin = 8
    line_height = 20  # 每行高度
    gap = 26  # 两行之间的间隔

    def _wrap_text(text, max_width):
        """将文本按最大宽度换行"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _get_text_dimensions(text_lines):
        """计算多行文本的总尺寸"""
        max_width = 0
        total_height = 0

        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            max_width = max(max_width, width)
            total_height += line_height

        return max_width, total_height
    # 计算文本包围盒
    def _text_bbox(text):
        # textbbox 返回 (x0, y0, x1, y1)
        return draw.textbbox((0, 0), text, font=font)

        # 计算图像宽度作为最大文本宽度（留出边距）

    img_width = img_np.shape[1]
    max_text_width = img_width - 2 * margin

    # 换行处理
    action_lines = _wrap_text(text_action, max_text_width)
    eef_lines = _wrap_text(text_eef, max_text_width)

    # 计算文本框尺寸
    action_width, action_height = _get_text_dimensions(action_lines)
    eef_width, eef_height = _get_text_dimensions(eef_lines)

    box_w = max(action_width, eef_width) + margin * 2
    box_h = action_height + eef_height + gap + margin * 2

    # 半透明背景
    overlay_rect = (0, 0, box_w, box_h)
    draw.rectangle(overlay_rect, fill=(0, 0, 0, 50))

    # 绘制action文本（多行）
    y = margin
    for line in action_lines:
        draw.text((margin, y), line, fill=color_action, font=font)
        y += line_height

    # 绘制eef文本（多行）
    y += gap
    for line in eef_lines:
        draw.text((margin, y), line, fill=color_eef, font=font)
        y += line_height

    return np.asarray(pil_img)


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
        # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        for episode_idx in tqdm.tqdm(range(3)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False  # 防止异常情况下未定义

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

                    # 规划 / 取动作
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
                    # print(f"Raw action range: min={np.min(action):.3f}, max={np.max(action):.3f}")

                    # === 在“当前帧”上叠加 action & eef_pos，并保存到缓存 ===
                    annotated = _overlay_text_on_image(img, action, obs["robot0_eef_pos"])
                    replay_images.append(annotated)

                    # 执行动作，推进环境
                    obs, reward, done, info = env.step(action.tolist())
                    env.env.render()




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

            # === 将缓存的每一帧保存为单张图片 ===
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            out_dir = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_ep{episode_idx+1:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            import cv2


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
