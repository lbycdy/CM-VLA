import os
import time
from datetime import datetime

import numpy as np
import glfw

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv

os.environ["MUJOCO_GL"] = "egl"

# =========================
# 0) 录制保存配置
# =========================
# 保存根目录（会自动创建）
SAVE_ROOT = os.path.join(os.getcwd(), "recordings", datetime.now().strftime("%Y%m%d_%H%M%S"))
RGB_DIR = os.path.join(SAVE_ROOT, "rgb")
DEPTH_DIR = os.path.join(SAVE_ROOT, "depth")
EEF_DIR = os.path.join(SAVE_ROOT, "eefpose")
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(EEF_DIR, exist_ok=True)

# 只在你“有输入动作”的步保存（和你打印动作那一步一致）。
# 如果你想连 0 动作也保存，改成 False。
SAVE_ONLY_WHEN_INPUT = True

# 有的 mujoco 图像会上下颠倒；你如果发现保存的 png 是倒的，把它改成 False
FLIP_UD = True

# 目标相机（rgb / depth 都按这个列表保存）
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]

# 深度保存方式："flat"=一串数字（每行一个数）；"matrix"=二维矩阵文本
DEPTH_SAVE_MODE = "flat"

# -------------------------
# png 保存（优先 PIL，缺失就走 imageio）
# -------------------------
try:
    from PIL import Image
    def _save_png(path, rgb_uint8):
        Image.fromarray(rgb_uint8).save(path)
except Exception:
    import imageio.v2 as imageio
    def _save_png(path, rgb_uint8):
        imageio.imwrite(path, rgb_uint8)

def _extract_eef_pos(obs, sim=None):
    if isinstance(obs, dict):
        for k in ("robot0_eef_pos", "eef_pos", "eef_position"):
            if k in obs:
                return np.asarray(obs[k], dtype=float).reshape(3,)
    if sim is not None:
        for site in ("gripper0_grip_site", "robot0_grip_site", "robot0_gripper_site", "eef_site"):
            try:
                return np.asarray(sim.data.get_site_xpos(site), dtype=float).reshape(3,)
            except Exception:
                pass
    return np.full(3, np.nan, dtype=float)

def _extract_eef_quat(obs, sim=None):
    if isinstance(obs, dict):
        for k in ("robot0_eef_quat", "eef_quat", "eef_orientation"):
            if k in obs:
                return np.asarray(obs[k], dtype=float).reshape(4,)
    return np.full(4, np.nan, dtype=float)

def _save_depth_txt(path, depth):
    depth = np.asarray(depth, dtype=float)
    if DEPTH_SAVE_MODE == "matrix":
        np.savetxt(path, depth, fmt="%.8f")
    else:
        flat = depth.reshape(-1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# shape {depth.shape[0]} {depth.shape[1]}\n")
            for v in flat:
                f.write(f"{v:.8f}\n")

def _get_obs_key(obs, cam, suffix):
    """兼容不同命名：优先 cam_{suffix}，找不到就模糊匹配。"""
    k = f"{cam}_{suffix}"
    if k in obs:
        return k
    for kk in obs.keys():
        if kk.endswith(f"_{suffix}") and cam in kk:
            return kk
    return None

def save_step_artifacts(obs, sim, step_idx, action):
    """
    保存：
      - 每个相机 rgb -> png
      - 每个相机 depth -> txt（数字串）
      - eef pose -> txt（[x y z qx qy qz qw]）
    """
    # 1) EEF pose
    pos = _extract_eef_pos(obs, sim)
    quat = _extract_eef_quat(obs, sim)
    pose = np.concatenate([pos, quat], axis=0)
    pose_path = os.path.join(EEF_DIR, f"eefpose_{step_idx:06d}.txt")
    np.savetxt(pose_path, pose.reshape(1, -1), fmt="%.8f")

    # 2) 每个相机的 RGB / Depth
    for cam in CAMERA_NAMES:
        rgb_key = _get_obs_key(obs, cam, "image")
        depth_key = _get_obs_key(obs, cam, "depth")

        if rgb_key is not None:
            rgb = np.asarray(obs[rgb_key])
            if FLIP_UD:
                rgb = np.flipud(rgb)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb_path = os.path.join(RGB_DIR, f"{cam}_{step_idx:06d}.png")
            _save_png(rgb_path, rgb)

        if depth_key is not None:
            depth = np.asarray(obs[depth_key], dtype=float)
            if FLIP_UD:
                depth = np.flipud(depth)
            depth_path = os.path.join(DEPTH_DIR, f"{cam}_{step_idx:06d}.txt")
            _save_depth_txt(depth_path, depth)

# ========== 1) 初始化任务 ==========
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()
task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] task {task_id} from suite {task_suite_name}, language: {task.language}")

# ========== 2) 初始化环境 ==========
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
    "use_camera_obs": True,
    "has_offscreen_renderer": True,
    "camera_names": CAMERA_NAMES,
    "camera_depths": [True] * len(CAMERA_NAMES),
    "render_camera": "frontview",
}
env = OnScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

sim = env.env.sim

print("\n=== 录制输出目录 ===")
print(SAVE_ROOT)

# ========== 3) 初始化 GLFW 键盘控制 ==========
if not glfw.init():
    raise Exception("无法初始化 GLFW（可能没有图形界面）")

glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
keyboard_window = glfw.create_window(200, 100, "Keyboard Control", None, None)
glfw.make_context_current(keyboard_window)

# action: [dx, dy, dz, dRx, dRy, dRz, gripper]
action = np.zeros(7)

step_size = 1
rot_step = 0.05

print("\n=== 实时控制说明 ===")
print("W/S: 前后 | A/D: 左右 | R/F: 上下")
print("J/L/I/K/U/O: 旋转 | Z/X: 合拢/张开 | Q: 退出\n")

step_idx = 0

# ========== 4) 控制循环 ==========
try:
    while not glfw.window_should_close(keyboard_window):
        glfw.poll_events()

        # 用于判断“这一步是否有输入”
        action_copy = action.copy()

        # 平移控制
        if glfw.get_key(keyboard_window, glfw.KEY_W) == glfw.PRESS:
            action[1] += step_size
        if glfw.get_key(keyboard_window, glfw.KEY_S) == glfw.PRESS:
            action[1] -= step_size
        if glfw.get_key(keyboard_window, glfw.KEY_A) == glfw.PRESS:
            action[0] -= step_size
        if glfw.get_key(keyboard_window, glfw.KEY_D) == glfw.PRESS:
            action[0] += step_size
        if glfw.get_key(keyboard_window, glfw.KEY_R) == glfw.PRESS:
            action[2] += step_size
        if glfw.get_key(keyboard_window, glfw.KEY_F) == glfw.PRESS:
            action[2] -= step_size

        # 旋转控制
        if glfw.get_key(keyboard_window, glfw.KEY_J) == glfw.PRESS:
            action[3] -= rot_step
        if glfw.get_key(keyboard_window, glfw.KEY_L) == glfw.PRESS:
            action[3] += rot_step
        if glfw.get_key(keyboard_window, glfw.KEY_I) == glfw.PRESS:
            action[4] += rot_step
        if glfw.get_key(keyboard_window, glfw.KEY_K) == glfw.PRESS:
            action[4] -= rot_step
        if glfw.get_key(keyboard_window, glfw.KEY_U) == glfw.PRESS:
            action[5] += rot_step
        if glfw.get_key(keyboard_window, glfw.KEY_O) == glfw.PRESS:
            action[5] -= rot_step

        # 夹爪
        if glfw.get_key(keyboard_window, glfw.KEY_Z) == glfw.PRESS:
            action[6] = 1.0
        if glfw.get_key(keyboard_window, glfw.KEY_X) == glfw.PRESS:
            action[6] = -1.0

        # 退出
        if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
            print("退出控制循环。")
            break

        obs, reward, done, info = env.step(action)

        # 是否保存：默认只在“你确实输入动作”的那步保存
        has_input = not np.array_equal(action, action_copy)
        if (not SAVE_ONLY_WHEN_INPUT) or has_input:
            step_idx += 1
            save_step_artifacts(obs, sim, step_idx, action)

            # 可选：简单打印一下 sanity check
            eef_pos = _extract_eef_pos(obs, sim)
            print(f"[save {step_idx:06d}] action={np.array2string(action, precision=3, suppress_small=True)} eef={np.array2string(eef_pos, precision=4)}")

        env.env.render()

        # 清空 action（保持你的原逻辑）
        action = np.zeros(7)
        action[-1] = -1.0

finally:
    env.close()
    glfw.destroy_window(keyboard_window)
    glfw.terminate()
    print("仿真结束。")
