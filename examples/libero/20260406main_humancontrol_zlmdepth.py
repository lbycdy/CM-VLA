import os
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

# ✅ 关键：保存“米制深度（相机光轴 Z 深度）”
# robosuite / mujoco 的 obs["xxx_depth"] 通常是 z-buffer 归一化到 [0,1] 的值，
# 这里会用 near/far 把它线性化成 meters 后再保存。
SAVE_DEPTH_IN_METERS = True

# 深度图映射模式：
# - "auto": 自动在 zeronear / zerofar 中选择更合理的（推荐）
# - "zeronear": depth01[0]=near, depth01[1]=far
# - "zerofar" : depth01[1]=near, depth01[0]=far
DEPTH_MAP_MODE = "zeronear"  # ✅ 与 HuggingFace/libero_with_depth 一致：0=near, 1=far

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


# =========================
# 1) depth01 -> meters (camera Z depth)
# =========================
def _mujoco_near_far_meters(sim):
    """
    MuJoCo 的 vis.map.znear / zfar 是相对 model.stat.extent 的系数
    真正的距离（米）= extent * znear / zfar
    """
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return near, far


def _depth01_to_meters(depth01, sim, depth_map="zeronear"):
    """
    把 [0,1] 的 z-buffer 深度线性化成“相机坐标系下沿视线方向的 Z 深度”（米）。
    depth_map:
      - 'zeronear': 0=near, 1=far
      - 'zerofar' : 1=near, 0=far
    """
    near, far = _mujoco_near_far_meters(sim)
    z = np.asarray(depth01, dtype=np.float32)
    z = np.clip(z, 0.0, 1.0)
    if depth_map.lower() == "zerofar":
        z = 1.0 - z

    # 线性化（得到 camera Z depth, meters）
    depth_m = near / (1.0 - z * (1.0 - near / far))
    return depth_m, near, far


def _choose_depth_map_auto(depth01, sim):
    """
    自动在 zeronear / zerofar 中选择一个更合理的映射。
    策略：看线性化后的深度落在 [near, far] 区间内的像素比例（越大越合理）。
    """
    d_zn, near, far = _depth01_to_meters(depth01, sim, "zeronear")
    d_zf, _, _ = _depth01_to_meters(depth01, sim, "zerofar")

    def score(d):
        ok = np.isfinite(d) & (d >= near * 0.99) & (d <= far * 1.01)
        return float(ok.mean())

    s_zn = score(d_zn)
    s_zf = score(d_zf)

    if s_zf > s_zn:
        return "zerofar"
    if s_zn > s_zf:
        return "zeronear"
    # 打平：用“中位数不要贴近 near plane”做选择。
    # 直觉：如果映射搞反了，中位数会非常接近 near (例如 0.02~0.1m)，看起来就像你现在的 0.1。
    med_zn = float(np.nanmedian(d_zn))
    med_zf = float(np.nanmedian(d_zf))
    return "zeronear" if med_zn >= med_zf else "zerofar"


def _maybe_to_meters(depth, sim):
    """
    depth 可能是：
      - [0,1] 的归一化 z-buffer（最常见）
      - 已经是米制深度（少数封装可能会这样返回）
    这里做一个保守判断：若 max <= 1.02 认为是 z-buffer，做线性化；否则视为米。
    """
    depth = np.asarray(depth)
    if depth.size == 0:
        return depth, None

    dmax = float(np.nanmax(depth))
    if dmax <= 1.02:
        mode = DEPTH_MAP_MODE
        if mode == "auto":
            mode = _choose_depth_map_auto(depth, sim)
        depth_m, near, far = _depth01_to_meters(depth, sim, mode)
        return depth_m, {"unit": "meters", "near": near, "far": far, "depth_map": mode}
    else:
        # 看起来已经不是 [0,1]，按“已是米制”处理
        return depth.astype(np.float32), {"unit": "meters", "near": None, "far": None, "depth_map": "already_meters"}


def _save_depth_txt(path, depth, meta=None):
    """保存深度为“纯数字序列”（每行一个 float），不写 header，方便对齐 HF 数据集的 sequence 格式。
    如果你还想保留 near/far/depth_map 等信息，可以把 meta 单独存到 json / txt。
    """
    depth = np.asarray(depth, dtype=np.float32)
    if DEPTH_SAVE_MODE == "matrix":
        np.savetxt(path, depth, fmt="%.8f")
        return
    flat = depth.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        for v in flat:
            f.write(f"{float(v):.8f}\n")


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
      - 每个相机 depth -> txt（数字串），默认保存“米制 Z 深度”
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
            depth = np.asarray(obs[depth_key], dtype=np.float32)
            if FLIP_UD:
                depth = np.flipud(depth)

            meta = None
            if SAVE_DEPTH_IN_METERS:
                depth, meta = _maybe_to_meters(depth, sim)
            # debug：检查是否真的是“米制深度”（可注释掉）
            if meta is not None and meta.get('unit') == 'meters':
                dmin = float(np.nanmin(depth)); dmed = float(np.nanmedian(depth)); dmax = float(np.nanmax(depth))
                print(f"    [{cam}] depth(m) min/med/max = {dmin:.4f} / {dmed:.4f} / {dmax:.4f} | near={meta.get('near'):.4f} far={meta.get('far'):.1f} map={meta.get('depth_map')}")

            # debug：检查是否真的是“米制深度”
            if meta is not None and meta.get('unit') == 'meters':
                dmin = float(np.nanmin(depth)); dmed = float(np.nanmedian(depth)); dmax = float(np.nanmax(depth))

            depth_path = os.path.join(DEPTH_DIR, f"{cam}_{step_idx:06d}.txt")
            _save_depth_txt(depth_path, depth, meta=meta)




def _write_run_meta(run_dir, sim):
    """写一份 run 级别的 meta：MuJoCo 近远裁剪面（米）等。"""
    try:
        near = float(sim.model.vis.map.znear * sim.model.stat.extent)
        far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
    except Exception:
        near, far = None, None
    meta = {
        "depth_map_mode": DEPTH_MAP_MODE,
        "mujoco_near_m": near,
        "mujoco_far_m": far,
        "camera_names": list(CAMERA_NAMES),
        "flip_ud": bool(FLIP_UD),
    }
    meta_path = os.path.join(run_dir, "run_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    print("[info] wrote meta:", meta_path)

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
_write_run_meta(SAVE_ROOT, sim)

print("\n=== 录制输出目录 ===")
print(SAVE_ROOT)

# 打印一下 near/far（方便你确认尺度）
try:
    near, far = _mujoco_near_far_meters(sim)
    print(f"[depth] mujoco near={near:.6f}m far={far:.6f}m  (DEPTH_MAP_MODE={DEPTH_MAP_MODE})")
except Exception as e:
    print("[depth] failed to read near/far:", e)

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