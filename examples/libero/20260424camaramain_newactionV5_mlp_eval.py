#export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/third_party/libero"
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
import glob
import re
import pandas as pd
import json
from io import BytesIO
from typing import Dict, List, Tuple, Any
from PIL import Image


# import pandas as pd  # ← 新增：读取 parquet 作为 GT



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

import numpy as np






@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 1

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 1 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_20251219actionV5"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # === 新增：GT 比较相关参数 ===
    #################################################################################################################
    data_root: str = os.path.expanduser("~/.cache/huggingface/lerobot/lbycdy/libero_20251219actionV5/data")
    compare_gt: bool = True              # 是否对比标签动作
    gt_frame: str = "base"               # "base" -> 比对 parquet['actions']；"camera" -> 比对 parquet['actions']
    align_skip: int = 0                  # 若需要跳过 GT 前若干步用于对齐，这里设置（默认 0）
    save_eval_csv: str = "action_eval_log.csv"  # 汇总保存路径
    offline_eval: bool = True
    compare_space: str = "cam"  # "cam" or "base"
    infer_every_step: bool = True  # teacher-forcing: 每帧都 infer 一次
    flip_images_180: bool = True  # 是否对图像做 180° 翻转（与你在线仿真 eval 的预处理一致）  # True: 读 parquet 做 teacher-forcing 对比; False: 跑仿真 rollout
    use_actions_base_gt: bool = False  # True: 直接用数据集里的 actions_base 做 GT（若存在）；False: actions(相机系) -> MLP -> base

def _normalize_task_text(s: str) -> str:
    # task 文本匹配时，把多余空格规整一下，避免 “两个空格/换行” 导致匹配失败
    return " ".join(str(s).strip().split())

def _resolve_lerobot_root(data_root: str) -> pathlib.Path:
    """
    physical-intelligence/libero 的 data_root 既可能是数据集根目录，也可能直接指向 data/。
    这里统一返回数据集根目录（包含 meta/ 和 data/）。
    """
    p = pathlib.Path(os.path.expanduser(data_root))
    if p.name == "data" and (p.parent / "meta").exists():
        return p.parent
    if (p / "meta").exists():
        return p
    # fallback: 如果传的是根目录但没有 meta（或你本地结构不同），也继续用它
    return p

def _resolve_lerobot_data_dir(data_root: str) -> pathlib.Path:
    root = _resolve_lerobot_root(data_root)
    d = root / "data"
    return d if d.exists() else root

def _load_tasks_jsonl(dataset_root: pathlib.Path) -> Dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        logging.warning(f"[OFFLINE] meta/tasks.jsonl not found under {dataset_root}; will fallback to task_id==task_index.")
        return {}
    out: Dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("task_index", obj.get("index", obj.get("id")))
            txt = obj.get("task", obj.get("text", obj.get("instruction")))
            if idx is None or txt is None:
                continue
            out[int(idx)] = str(txt)
    return out

def _map_suite_taskid_to_dataset_taskindex(task_suite, dataset_tasks: Dict[int, str]) -> Dict[int, int]:
    """把 LIBERO suite 内的 task_id 映射到 LeRobot 数据集里的 task_index。"""
    if not dataset_tasks:
        return {i: i for i in range(task_suite.n_tasks)}
    inv = {_normalize_task_text(v): k for k, v in dataset_tasks.items()}
    out: Dict[int, int] = {}
    for i in range(task_suite.n_tasks):
        key = _normalize_task_text(task_suite.get_task(i).language)
        if key in inv:
            out[i] = int(inv[key])
        else:
            out[i] = i
            logging.warning(f"[OFFLINE] Cannot match suite task_id={i} to dataset task_index via text. Fallback: task_index={i}.")
    return out

def _index_parquets_by_task_index(dataset_root: pathlib.Path) -> Dict[int, List[pathlib.Path]]:
    """
    返回: task_index -> [episode parquet paths]。
    优先用 meta/episodes.jsonl（更快），没有则 fallback 扫 parquet 首行 task_index。
    """
    data_dir = _resolve_lerobot_data_dir(str(dataset_root))

    # 先建立 episode_index -> parquet path（靠文件名正则）
    ep2path: Dict[int, pathlib.Path] = {}
    for p in sorted(data_dir.rglob("*.parquet")):
        m = re.search(r"episode_(\d+)\.parquet$", p.name)
        if m:
            ep2path[int(m.group(1))] = p

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.exists() and ep2path:
        task2paths: Dict[int, List[pathlib.Path]] = {}
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ep = obj.get("episode_index", obj.get("episode_id", obj.get("index")))
                ti = obj.get("task_index")
                if ti is None and isinstance(obj.get("task"), dict):
                    ti = obj["task"].get("task_index")
                if ep is None or ti is None:
                    continue
                pth = ep2path.get(int(ep))
                if pth is not None:
                    task2paths.setdefault(int(ti), []).append(pth)
        if task2paths:
            return task2paths

    # fallback: 扫 parquet（只读 task_index 列）
    task2paths: Dict[int, List[pathlib.Path]] = {}
    for p in tqdm.tqdm(sorted(data_dir.rglob("*.parquet")), desc="Indexing parquets (read task_index)"):
        df0 = pd.read_parquet(p, columns=["task_index"], engine="pyarrow")
        ti = int(df0["task_index"].iloc[0])
        task2paths.setdefault(ti, []).append(p)
    return task2paths

def _decode_lerobot_image(v: Any, dataset_root: pathlib.Path) -> np.ndarray:
    """LeRobot 官方 libreo parquet 的 image/wrist_image 是 struct<bytes,path>。这里解码成 HxWx3 uint8 numpy。"""
    if isinstance(v, np.ndarray):
        return v
    # pandas/pyarrow 常见表示：dict {'bytes':..., 'path':...}
    b = None
    p = None
    if isinstance(v, dict):
        b = v.get("bytes")
        p = v.get("path")
    elif isinstance(v, (tuple, list)) and len(v) == 2:
        b, p = v[0], v[1]
    else:
        # 兜底：尝试属性访问
        b = getattr(v, "bytes", None)
        p = getattr(v, "path", None)

    if b is not None and len(b) != 0:
        if isinstance(b, memoryview):
            b = b.tobytes()
        img = Image.open(BytesIO(b)).convert("RGB")
        return np.asarray(img)

    if p:
        img_path = pathlib.Path(str(p))
        if not img_path.is_absolute():
            img_path = dataset_root / img_path
        img = Image.open(img_path).convert("RGB")
        return np.asarray(img)

    raise ValueError("Cannot decode LeRobot image struct")
def _guess_obs_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    # 尽量兼容不同 LeRobot 版本的列名
    img_candidates = [
        "observation.images.image",
        "observation.image",
        "image",
    ]
    wrist_candidates = [
        "observation.images.wrist_image",
        "observation.wrist_image",
        "wrist_image",
    ]
    state_candidates = [
        "observation.state",
        "state",
    ]

    def pick(cands: List[str], kind: str) -> str:
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"Cannot find {kind} column. Available columns (first 30): {list(df.columns)[:30]}")

    return pick(img_candidates, "image"), pick(wrist_candidates, "wrist_image"), pick(state_candidates, "state")

def _as_numpy(x):
    # parquet 里有时会是 list / np.ndarray / object；统一成 np.ndarray
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)

def eval_libero_offline_from_parquet(args: Args) -> None:
    np.random.seed(args.seed)

    # 初始化 LIBERO suite（只是用它的 task.language 文本来对齐数据，不跑仿真）
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"[OFFLINE] Task suite: {args.task_suite_name}")

    # max_steps 沿用你原来的设置逻辑 :contentReference[oaicite:6]{index=6}
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    action_mapper = None
    if args.compare_space.lower() == "base":
        action_mapper = ActionMappingInference(
            model_path="/home/lbycdy/work/camerapi/best_action_mapping.pth",
            device="cuda",
        )

    dataset_root = _resolve_lerobot_root(args.data_root)
    dataset_tasks = _load_tasks_jsonl(dataset_root)
    suite2taskidx = _map_suite_taskid_to_dataset_taskindex(task_suite, dataset_tasks)
    taskidx2paths = _index_parquets_by_task_index(dataset_root)

    eval_rows = []
    missing = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        dataset_task_index = suite2taskidx.get(task_id, task_id)

        ep_paths = taskidx2paths.get(dataset_task_index, [])
        if not ep_paths:
            missing += 1
            logging.warning(f"[OFFLINE] No parquet episodes found for task_id={task_id} (task_index={dataset_task_index})")
            continue

        # 取前 num_trials_per_task 条 episode 做对比
        ep_paths = ep_paths[: args.num_trials_per_task]

        for epi, ep_path in enumerate(ep_paths):
            df = pd.read_parquet(ep_path, columns=["image","wrist_image","state","actions","actions_base","task_index","episode_index","frame_index"], engine="pyarrow")
            # prompt 用 parquet 里的 task（与数据严格一致）
            prompt = dataset_tasks.get(dataset_task_index, task.language)

            img_col, wrist_col, state_col = _guess_obs_columns(df)

            # GT actions（相机系）-> base
            gt_actions_cam = np.stack([_as_numpy(a).astype(np.float32) for a in df["actions"].to_numpy()])
            # 可选对齐跳过
            start_t = max(0, int(args.align_skip))

                        # 统计误差（teacher-forcing：每一帧用 parquet 的 obs 推理，然后和同一帧的标签 actions 做对比）
            errs = []  # list of (7,)

            T = min(len(df), max_steps)
            for t in range(start_t, T):
                # 从 parquet 取 obs
                img = _decode_lerobot_image(df[img_col].iloc[t], dataset_root)
                wrist = _decode_lerobot_image(df[wrist_col].iloc[t], dataset_root)
                state = _as_numpy(df[state_col].iloc[t]).astype(np.float32)

                # 图像预处理：可选 180° 翻转 + resize_with_pad + uint8
                if args.flip_images_180:
                    img = np.ascontiguousarray(img[::-1, ::-1])
                    wrist = np.ascontiguousarray(wrist[::-1, ::-1])

                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": state,
                    "prompt": str(prompt),
                }

                # 推理：policy 输出的是 camera-frame actions chunk，取第一步与标签对齐
                pred_chunk_cam = client.infer(element)["actions"]
                pred_cam = np.asarray(pred_chunk_cam[0], dtype=np.float32)

                if args.compare_space.lower() == "cam":
                    # 直接在 camera frame 比：GT = parquet["actions"]
                    gt = np.asarray(gt_actions_cam[t], dtype=np.float32)
                    pred = pred_cam
                elif args.compare_space.lower() == "base":
                    # 在 base frame 比：pred/gt 都 cam->base（或用 actions_base 当 GT）
                    assert action_mapper is not None, "compare_space='base' requires action_mapper"
                    pred = np.asarray(action_mapper.cam_to_base(pred_cam), dtype=np.float32)
                    if args.use_actions_base_gt and "actions_base" in df.columns:
                        gt = np.asarray(_as_numpy(df["actions_base"].iloc[t]), dtype=np.float32)
                    else:
                        gt = np.asarray(action_mapper.cam_to_base(gt_actions_cam[t]), dtype=np.float32)
                else:
                    raise ValueError(f"Unknown compare_space={args.compare_space}. Use 'cam' or 'base'.")

                errs.append(pred - gt)

            if not errs:
                logging.warning(f"[OFFLINE] Empty episode? {ep_path}")
                continue

            err = np.stack(errs)  # (T,7)
            mae_per_dim = np.mean(np.abs(err), axis=0)
            rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=0))
            l2 = np.mean(np.linalg.norm(err, axis=1))

            row = {
                "task_id": task_id,
                "episode_rank": epi,
                "parquet": str(ep_path),
                "steps": int(err.shape[0]),
                "mae_mean": float(np.mean(mae_per_dim)),
                "rmse_mean": float(np.mean(rmse_per_dim)),
                "l2_mean": float(l2),
                # 拆开看 xyz / rpy / gripper
                "mae_xyz": float(np.mean(mae_per_dim[:3])),
                "mae_rpy": float(np.mean(mae_per_dim[3:6])),
                "mae_grip": float(mae_per_dim[6]),
            }
            eval_rows.append(row)

            logging.info(
                f"[OFFLINE] task_id={task_id} epi={epi} steps={row['steps']} "
                f"MAE={row['mae_mean']:.4f} RMSE={row['rmse_mean']:.4f} L2={row['l2_mean']:.4f} "
                f"(xyz={row['mae_xyz']:.4f}, rpy={row['mae_rpy']:.4f}, g={row['mae_grip']:.4f})"
            )

    # 保存汇总 CSV
    if eval_rows:
        out = pathlib.Path(args.save_eval_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(eval_rows).to_csv(out, index=False)
        logging.info(f"[OFFLINE] Wrote summary CSV to: {out.resolve()}")

    if missing:
        logging.warning(f"[OFFLINE] {missing}/{num_tasks_in_suite} tasks had no matching parquets. "
                        f"Usually this means task 文本不完全一致（可打印 parquet['task'][0] 对照一下）。")

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

                        action_chunk = client.infer(element)["actions"]

                        print(f"[DEBUG] 原始相机坐标系动作: {action_chunk[0]}")

                        # === 使用神经网络转换动作 ===
                        converted_action_chunk = []
                        for actions in action_chunk:
                            # 直接调用推理器进行转换
                            actions_base = action_mapper.cam_to_base(actions)
                            converted_action_chunk.append(actions_base)


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


def main(args: Args) -> None:
    if args.offline_eval:
        eval_libero_offline_from_parquet(args)
    else:
        eval_libero(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)