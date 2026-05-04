# examples/libero/camaramain_testdataset_zlm_meta_text.py
import os
import glob
import json
import re
import logging
import pathlib
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv

# ========== 基础设置 ==========
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("libero-replay")

# 是否需要窗口渲染（会走 cv2.imshow 路径）
SHOW_WINDOW = True  # ← 打开，与代码2的“显示到屏幕”一致

# 渲染后端：本地桌面可用 "glfw"，服务器无头可换 "osmesa"
os.environ["MUJOCO_GL"] = "glfw"
# 与代码2一致：在本地桌面设置 DISPLAY
os.environ["DISPLAY"] = ":0"

# ========== 数据目录（按需修改） ==========
DATA_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/lbycdy/libero_camera/data")
META_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/lbycdy/libero_camera/meta")
TASKS_JSONL = os.path.join(META_DIR, "tasks.jsonl")


# ---- 增量换帧：cam -> base （旋转-only）----
def cam_delta_to_base_delta(actions7: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    """
    actions7: [dx,dy,dz, dthx,dthy,dthz, g] (CAMERA frame delta)
    T_cam_to_base: 4x4 齐次变换矩阵，表示从 camera 到 base 的变换
    return: same 7D delta in BASE frame
    """
    T_cam_to_base = get_T_cam_to_base(T_cam_to_base)
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

def get_T_cam_to_base(task_suite_name: str):
    # 这里按常见 LIBERO 套件给定相机到基座的外参；如你的项目有更准的标定，替换即可
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
    return np.eye(4)



# ========== 读取数据集 meta 映射（严格要求存在） ==========
def load_task_mapping(tasks_jsonl_path: str) -> Dict[int, str]:
    """
    从数据集自带的 meta/tasks.jsonl 读取 {task_index(int): task_text(str)}。
    允许键名是 task/language/instruction/language_instruction/text 之一。
    """
    if not os.path.exists(tasks_jsonl_path):
        raise FileNotFoundError(
            f"未找到数据集的任务映射文件：{tasks_jsonl_path}\n"
            "请确认你的数据集包含 meta/tasks.jsonl；每行形如："
            '{"task_index": 0, "language": "pick up the ..."}'
        )
    mapping: Dict[int, str] = {}
    bad_lines: List[str] = []
    with open(tasks_jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_lines.append(f"第{ln}行 JSON 解析失败：{s[:120]}")
                continue
            idx = obj.get("task_index")
            text = (obj.get("task") or obj.get("language") or obj.get("instruction")
                    or obj.get("language_instruction") or obj.get("text"))
            if isinstance(idx, (int, np.integer)) and isinstance(text, str) and text.strip():
                mapping[int(idx)] = text.strip()
            else:
                bad_lines.append(f"第{ln}行缺少 task_index 或 文本键(task/language/...)：{s[:120]}")
    if not mapping:
        raise RuntimeError(
            "tasks.jsonl 中未解析出任何 (task_index → 文本) 映射。\n" +
            ("\n".join(bad_lines[:10]) if bad_lines else "文件内容可能不符合预期。")
        )
    logger.info(f"已加载任务映射 {len(mapping)} 条，源文件：{tasks_jsonl_path}")
    return mapping

# ========== 用“文本任务名”解析到 LIBERO 任务 ==========
def canon(s: str) -> str:
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set_jaccard(a: str, b: str) -> float:
    A, B = set(canon(a).split()), set(canon(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def resolve_task_by_name(task_name: str, min_sim: float = 0.6) -> Tuple[str, int, object]:
    """
    给定数据集里的任务文本，在所有 LIBERO 套件中寻找匹配任务。
    - 先做规范化后的完全相等；
    - 不行则用 token Jaccard 相似度，阈值默认 0.6；
    - 仍不行则报错并给出若干最相近候选，便于排查。
    返回 (suite_name, task_index, task_obj)。
    """
    bd = benchmark.get_benchmark_dict()
    exact = None
    best = (0.0, None)  # (score, (suite, idx, task))
    c_query = canon(task_name)

    for suite_name, suite_cls in bd.items():
        try:
            suite = suite_cls()
        except Exception:
            continue
        for idx in range(suite.n_tasks):
            try:
                t = suite.get_task(idx)
                t_lang = t.language
            except Exception:
                continue
            c_lang = canon(t_lang)
            if c_lang == c_query:
                exact = (suite_name, idx, t)
                break
            score = token_set_jaccard(task_name, t_lang)
            if score > best[0]:
                best = (score, (suite_name, idx, t))
        if exact:
            break

    if exact:
        return exact

    # 提示最相近的三个，帮助定位措辞差异
    # 如果 best 分数不足阈值，报错
    top_candidates: List[Tuple[float, str, int, str]] = []
    for suite_name, suite_cls in bd.items():
        try:
            suite = suite_cls()
        except Exception:
            continue
        for idx in range(suite.n_tasks):
            try:
                t = suite.get_task(idx)
                s = token_set_jaccard(task_name, t.language)
                top_candidates.append((s, suite_name, idx, t.language))
            except Exception:
                continue
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    msg = [
        f"无法按文本任务名匹配到 LIBERO 任务：{task_name!r}",
        f"最高相似度={best[0]:.3f}（阈值={min_sim}），前3个候选："
    ]
    for s, sn, idx, lang in top_candidates[:3]:
        msg.append(f"  - sim={s:.3f}  [{sn}#{idx}]  {lang}")

    if best[0] >= min_sim and best[1] is not None:
        sn, idx, t = best[1]
        logger.warning("\n".join(msg))
        logger.warning("使用最优相似匹配作为近似结果。")
        return sn, idx, t

    raise RuntimeError("\n".join(msg))

# ========== 主流程 ==========
def build_env_from_task(task) -> OnScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    logger.info(f"BDDL 路径: {bddl}")
    env = OnScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)
    env.reset()  # 触发实际构建
    return env

def main1():
    # 1) 读取 episode 列表
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk-*/*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"❌ 未找到任何 parquet 文件: {DATA_DIR}")
    logger.info(f"发现 {len(parquet_files)} 个 episode 文件")

    # 2) 加载数据集 meta 映射（严格要求存在）
    idx2text = load_task_mapping(TASKS_JSONL)

    # 3) 用第一个 episode 的 task_index → 文本 → 匹配 LIBERO 任务 → 构建环境
    first_df = pd.read_parquet(parquet_files[0])
    if "task_index" not in first_df.columns:
        raise RuntimeError(f"{parquet_files[0]} 不包含 'task_index' 列，无法从 meta 映射到文本任务名")
    first_idx = int(first_df["task_index"].iloc[0])
    if first_idx not in idx2text:
        raise KeyError(
            f"tasks.jsonl 中没有 task_index={first_idx} 的映射。"
            f"可用键：{sorted(list(idx2text.keys()))[:10]} ..."
        )
    first_text = idx2text[first_idx]
    logger.info(f"首个 episode: task_index={first_idx}, 文本任务名={first_text!r}")

    suite_name, task_index, task = resolve_task_by_name(first_text)
    logger.info(f"将使用任务: [{suite_name}#{task_index}] {task.language}")

    env = build_env_from_task(task)
    current_key = (suite_name, task_index)

    # 4) 回放全部 episodes：每个 episode 用自己的 task_index 从 meta 取文本，再解析并（必要时）切换环境
    success_count = 0
    for file in tqdm.tqdm(parquet_files, desc="回放任务"):
        df = pd.read_parquet(file)
        if "task_index" not in df.columns:
            raise RuntimeError(f"{os.path.basename(file)} 缺少 'task_index' 列")

        ep_idx = int(df["task_index"].iloc[0])
        if ep_idx not in idx2text:
            raise KeyError(
                f"{os.path.basename(file)} 的 task_index={ep_idx} 在 tasks.jsonl 中没有对应文本。"
            )
        ep_text = idx2text[ep_idx]
        ep_suite, ep_task_index, ep_task = resolve_task_by_name(ep_text)

        if (ep_suite, ep_task_index) != current_key:
            logger.info(f"[切换任务] [{ep_suite}#{ep_task_index}] {ep_task.language}")
            env.close()
            env = build_env_from_task(ep_task)
            current_key = (ep_suite, ep_task_index)

        obs = env.reset()
        logger.info(f"开始执行: {os.path.basename(file)} 共 {len(df)} 步")
        logger.info(f"任务描述: {ep_task.language}")

        done = False
        for i, row in enumerate(df.itertuples(index=False)):
            # 优先用数据集中已有的基座系 actions；没有就把 action_cam 转换成基座系
            if hasattr(row, "actions") and row.actions is not None:
                # action_vec = np.array(row.actions, dtype=np.float32)
                action_vec = cam_delta_to_base_delta(row.actions, ep_suite)
            else:
                raise RuntimeError("既没有 'actions' 也没有 'action_cam'，无法执行动作")

            obs, reward, done, info = env.step(action_vec.tolist())

            # 与代码2一致：每步进行窗口渲染（需 GUI 环境）
            if SHOW_WINDOW:
                env.env.render()

            eef = obs.get("robot0_eef_pos", None)
            if eef is not None:
                print(f"[{i:04d}] reward={reward:.3f}, done={done}, eef_pos={eef}")
            else:
                print(f"[{i:04d}] reward={reward:.3f}, done={done}")

            if done:
                success_count += 1
                break

    logger.info(f"✅ 任务完成 {success_count}/{len(parquet_files)} episodes")
    env.close()

if __name__ == "__main__":
    main1()
