import collections
import dataclasses
import logging
import math
import pathlib
import os
import glob

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OnScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
# === NEW: offline compare needs these ===
import io, json
from PIL import Image

import pandas as pd  # ← 新增：读取 parquet 作为 GT

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

import numpy as np

# ---- 增量换帧：cam -> base （旋转-only）----
def cam_delta_to_base_delta(action_cam7: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    """
    action_cam7: [dx,dy,dz, dthx,dthy,dthz, g] (CAMERA frame delta)
    T_cam_to_base: 4x4 齐次变换矩阵，表示从 camera 到 base 的变换
    return: same 7D delta in BASE frame
    """
    action_cam7 = np.asarray(action_cam7, dtype=np.float32)
    assert action_cam7.shape[-1] == 7, f"action length must be 7, got {action_cam7.shape}"

    R_cb = T_cam_to_base[:3, :3]  # cam -> base 的旋转矩阵

    v_c = action_cam7[:3]
    w_c = action_cam7[3:6]
    g = action_cam7[6]

    # cam -> base
    v_b = R_cb @ v_c
    w_b = R_cb @ w_c

    return np.concatenate([v_b, w_b, [g]]).astype(np.float32)


# ========= Camera <-> Base Transformation =========
def get_T_cam_to_base(task_suite_name):
    matrices = {
        "libero_10": np.array([[0.0, 1.0, 0.0, 0.0],
                               [-0.258174524, 0.0, 0.966098295, 0.0552036108],
                               [0.966098295, 0.0, 0.258174524, -2.06578134],
                               [0.0, 0.0, 0.0, 1.0]]),
        "libero_spatial": np.array([[0.0, 1.0, 0.0, 0.0],
                                    [-0.258174524, 0.0, 0.966098295, -0.120174122],
                                    [0.966098295, 0.0, 0.258174524, -1.75036630],
                                    [0.0, 0.0, 0.0, 1.0]]),
        "libero_goal": np.array([[0.0, 1.0, 0.0, 0.0],
                                 [-0.258174524, 0.0, 0.966098295, -0.120174122],
                                 [0.966098295, 0.0, 0.258174524, -1.75036630],
                                 [0.0, 0.0, 0.0, 1.0]]),
        "libero_object": np.array([[0.0, 1.0, 0.0, 0.0],
                                   [-0.258174524, 0.0, 0.966098295, -0.214884654],
                                   [0.966098295, 0.0, 0.258174524, -1.71357071],
                                   [0.0, 0.0, 0.0, 1.0]])
    }

    for k in matrices:
        if task_suite_name.startswith(k):
            return matrices[k]
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _load_task_map(meta_dir: str):
    """读取 meta/tasks.jsonl，返回 {task_index:int -> task:str}"""
    path = os.path.join(meta_dir, "tasks.jsonl")
    task_map = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            task_map[int(obj["task_index"])] = obj["task"]
    return task_map

def _decode_image_field(field):
    """从 dataframe 的 image/wrist_image 字段解码成 np.uint8(H,W,3)"""
    img = Image.open(io.BytesIO(field["bytes"])).convert("RGB")
    return np.array(img, dtype=np.uint8)

def _preprocess_image(arr, size=224, rotate180=False):
    """与服务端一致的预处理：可选180°旋转 + resize_with_pad 到 size"""
    if rotate180:
        arr = arr[::-1, ::-1].copy()
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(arr, size, size))







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
    num_steps_wait: int = 5 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5  # Number of rollouts per task

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
    gt_frame: str = "base"               # "base" -> 比对 parquet['actions']；"camera" -> 比对 parquet['action_cam']
    align_skip: int = 0                  # 若需要跳过 GT 前若干步用于对齐，这里设置（默认 0）
    save_eval_csv: str = "action_eval_log.csv"  # 汇总保存路径
    # === NEW: offline（直接用数据集做 action_cam 对比）===
    compare_offline: bool = True          # True = 只做离线对比，不跑仿真
    offline_max_episodes: int = 2         # 最多对比多少个 episode（<=0 表示全部）
    offline_max_frames: int = 200         # 每个 episode 最多多少帧（<=0 表示全部）
    offline_print_first: int = 10         # 每个 episode 打印前多少帧详情
    rotate180: bool = False               # 如果你的数据转换阶段没做180°旋转，就保持 False

def compare_action_cam_offline(args: Args) -> None:
    """
    逐帧：用数据集里的 image + wrist_image + state + prompt 调服务端，
    拿预测的 action_cam，与标签 action_cam 直接比较（camera frame 内比较，不做外参变换）。
    """
    # 1) 数据文件列表
    data_dir = os.path.expanduser(os.path.join(os.path.dirname(os.path.dirname(args.data_root)), "libero_camera"))
    meta_dir = os.path.join(data_dir, "meta")
    data_root = os.path.join(data_dir, "data")
    chunks = sorted(glob.glob(os.path.join(data_root, "chunk-*")))
    if not chunks:
        raise FileNotFoundError(f"未找到数据目录: {data_root}/chunk-*")
    parquet_files = []
    for c in chunks:
        parquet_files.extend(sorted(glob.glob(os.path.join(c, "*.parquet"))))
    if args.offline_max_episodes > 0:
        parquet_files = parquet_files[: args.offline_max_episodes]
    logging.info(f"[offline] 将比较 {len(parquet_files)} 个 episode")

    # 2) 任务文本映射
    task_map = _load_task_map(meta_dir)

    # 3) 连接推理服务
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    all_pos_err, all_ang_err = [], []

    for p in tqdm.tqdm(parquet_files, desc="Compare episodes (camera frame)"):
        df = pd.read_parquet(p)
        ti = int(df["task_index"].iloc[0])
        ep = int(df["episode_index"].iloc[0])
        prompt = task_map.get(ti, f"<unknown task {ti}>")
        N = len(df) if args.offline_max_frames <= 0 else min(args.offline_max_frames, len(df))

        print(f"\n=== Episode: {os.path.basename(p)} | task_index={ti}, episode_index={ep}")
        print(f"Task: {prompt}")

        pos_errs, ang_errs = [], []

        for i in range(N):
            row = df.iloc[i]
            #gt_cam = np.asarray(row["action_cam"], dtype=np.float32)
            gt_cam = np.asarray(row["actions"], dtype=np.float32)

            img = _preprocess_image(_decode_image_field(row["image"]),
                                    size=args.resize_size, rotate180=args.rotate180)
            if "wrist_image" in df.columns:
                wrist = _preprocess_image(_decode_image_field(row["wrist_image"]),
                                          size=args.resize_size, rotate180=args.rotate180)
            else:
                wrist = img
            state = np.asarray(row["state"], dtype=np.float32)

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": prompt,
            }
            resp = client.infer(element)
            #pred_chunk = np.asarray(resp["action_cam"], dtype=np.float32)
            pred_chunk = np.asarray(resp["actions"], dtype=np.float32)
            pred_cam = pred_chunk[0, :7] if pred_chunk.ndim == 2 else pred_chunk[:7]

            pos_err = float(np.linalg.norm(pred_cam[:3] - gt_cam[:3], ord=2))
            ang_err = float(np.linalg.norm(pred_cam[3:6] - gt_cam[3:6], ord=2))
            pos_errs.append(pos_err); ang_errs.append(ang_err)

            if i < args.offline_print_first:
                print(f"[{i:03d}] "
                      f"gt_pos={gt_cam[:3]}, pred_pos={pred_cam[:3]}, |Δpos|={pos_err:.4f} | "
                      f"gt_ang={gt_cam[3:6]}, pred_ang={pred_cam[3:6]}, |Δang|={ang_err:.4f} | "
                      f"gt_grip={gt_cam[6]:+.3f}, pred_grip={pred_cam[6]:+.3f}")

        mep = np.mean(pos_errs) if pos_errs else np.nan
        mea = np.mean(ang_errs) if ang_errs else np.nan
        print(f"Episode mean |Δpos|={mep:.4e}, mean |Δang|={mea:.4e}  (frames={N})")

        all_pos_err.extend(pos_errs)
        all_ang_err.extend(ang_errs)

    if all_pos_err:
        print("\n======== Overall (camera frame) ========")
        print(f"Overall mean |Δpos|={np.mean(all_pos_err):.4e}, "
              f"mean |Δang|={np.mean(all_ang_err):.4e}, "
              f"median |Δpos|={np.median(all_pos_err):.4e}, "
              f"median |Δang|={np.median(all_ang_err):.4e}")
    else:
        print("没有有效帧被比较。")

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

    # === 新增：构建 (task_index, episode_index) -> parquet 路径 的索引 ===
    ep_index = build_episode_index(args.data_root) if args.compare_gt else {}

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
            obs = env.set_init_state(initial_states[episode_idx])

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
                        #action_chunk = client.infer(element)["action_cam"]
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action_cam = action_plan.popleft()
                    T_cam_to_base = get_T_cam_to_base(args.task_suite_name)

                    # Convert from camera frame to base frame
                    action_base = cam_delta_to_base_delta(action_cam, T_cam_to_base)



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

            # === 每个 episode 结束后：若开启 compare_gt，则读取 GT 并计算误差 ===
            if args.compare_gt:
                try:
                    # 取 GT：精确按 (task_id, episode_idx) 寻找
                    gt_actions = load_gt_actions(ep_index, task_id, episode_idx, frame=args.gt_frame)
                    # 对齐：忽略前 num_steps_wait 的等待步，只比较真实执行的预测动作
                    pred_arr = np.asarray(pred_actions_base, dtype=np.float32)
                    gt_arr   = gt_actions
                    if args.gt_frame == "base":
                        # gt 已经是基座系；无需转换
                        pass
                    else:
                        # 如果你想和相机系比，可以把 pred 转回相机系；这里为了简洁不做，通常推荐比较基座系
                        pass

                    # 可选跳过 GT 前若干步
                    if args.align_skip > 0:
                        gt_arr = gt_arr[args.align_skip:]

                    T = min(len(pred_arr), len(gt_arr))
                    if T <= 0:
                        raise RuntimeError("对齐后长度为 0，无法比较。")

                    pos_list, ang_list, grip_list, mse_list = [], [], [], []


                    row = {
                        "task_index": task_id,
                        "episode_index": episode_idx,
                        "T_compared": T,
                        "pos_L2_mean": float(np.mean(pos_list)),
                        "pos_L2_median": float(np.median(pos_list)),
                        "ori_deg_mean": float(np.mean(ang_list)),
                        "ori_deg_median": float(np.median(ang_list)),
                        "grip_abs_mean": float(np.mean(grip_list)),
                        "mse7_mean": float(np.mean(mse_list)),
                    }
                    eval_rows.append(row)
                    logging.info(f"[Eval GT] task={task_id:02d} ep={episode_idx:02d} "
                                 f"T={T} pos={row['pos_L2_mean']:.4f} "
                                 f"ori={row['ori_deg_mean']:.2f} grip={row['grip_abs_mean']:.4f} "
                                 f"mse7={row['mse7_mean']:.5f}")
                except Exception as e:
                    logging.warning(f"[Eval GT] 无法比较 task={task_id} ep={episode_idx}: {e}")

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




def build_episode_index(data_root: str):
    """
    构建 (task_index, episode_index) -> parquet 路径 的字典。
    要求每个 parquet 至少包含 task_index, episode_index 两列。
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据集目录不存在: {data_root}")
    files = sorted(glob.glob(os.path.join(data_root, "chunk-*", "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(f"在 {data_root}/chunk-*/ 下未找到 episode_*.parquet")
    index = {}
    for p in files:
        try:
            df = pd.read_parquet(p, columns=["task_index", "episode_index"])
            ti = int(df["task_index"].iloc[0])
            ei = int(df["episode_index"].iloc[0])
            index[(ti, ei)] = p
        except Exception as e:
            logging.warning(f"索引 {os.path.basename(p)} 失败：{e}")
    logging.info(f"[GT] 已索引 {len(index)} 个 episode")
    return index

def load_gt_actions(ep_index: dict, task_index: int, episode_index: int, frame: str = "base") -> np.ndarray:
    """
    读取指定 (task_index, episode_index) 的 GT 动作序列。
    frame="base" -> df['actions']；frame="camera" -> df['action_cam']。
    返回 [T,7] float32。
    """
    key = (task_index, episode_index)
    if key not in ep_index:
        raise KeyError(f"找不到 (task_index={task_index}, episode_index={episode_index}) 对应的 parquet")
    p = ep_index[key]
    df = pd.read_parquet(p)
    col = "actions" if frame == "base" else "action_cam"

    if col not in df.columns:
        raise KeyError(f"{os.path.basename(p)} 不含列 '{col}'")
    arr = np.stack(df[col].to_numpy()).astype(np.float32)
    if arr.shape[1] != 7:
        raise ValueError(f"{os.path.basename(p)} 的 '{col}' 形状应为 [T,7]，实际 {arr.shape}")
    return arr


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


def main(args: Args):
    if args.compare_offline:
        compare_action_cam_offline(args)
    else:
        eval_libero(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
