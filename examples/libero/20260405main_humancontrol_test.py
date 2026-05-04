import os
import time
import numpy as np
import glfw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv

os.environ["MUJOCO_GL"] = "egl"

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
    "render_camera": "frontview",
}
env = OnScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
low, high = env.env.action_spec
print("action_spec shape:", low.shape, high.shape)

init_states = task_suite.get_task_init_states(task_id)


# env.set_init_state(init_states[0])
# env.env.sim.forward()

# 只给 0 动作跑几步，观察 eef 是否自己在漂移/回位
# for i in range(30):
#     a = np.zeros(7)
#     a[-1] = -1.0  # 你的默认 gripper :contentReference[oaicite:2]{index=2}
#     obs, r, d, info = env.step(a)
#     if i in [0, 1, 2, 10, 29]:
#         print(i, obs["robot0_eef_pos"], obs["robot0_eef_quat"])

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

def _extract_arm_qpos(obs, sim=None, prefix="robot0_joint", dof=7):
    if isinstance(obs, dict):
        for k in ("robot0_joint_pos", "joint_pos", "arm_joint_pos"):
            if k in obs:
                arr = np.asarray(obs[k], dtype=float).flatten()
                if arr.size >= dof:
                    return arr[:dof]
    if sim is not None:
        q = []
        for i in range(1, dof + 1):
            try:
                q.append(float(sim.data.get_joint_qpos(f"{prefix}{i}")))
            except Exception:
                q.append(float("nan"))
        return np.asarray(q, dtype=float)
    return np.full(dof, np.nan, dtype=float)

# ================================================================
# === 手眼标定：计算相机 → 机械臂坐标系的变换矩阵 ==================
# ================================================================

def get_hand_eye_matrix(sim, cam_name="frontview", base_name="robot0_base"):
    """
    从 LIBERO/robosuite 仿真环境计算手眼标定外参矩阵：
      T_camera→robot = inv(T_world_camera) @ T_world_robot
    """

    # 相机在世界系下的位姿
    cam_pos = sim.data.get_camera_xpos(cam_name)
    cam_rot = sim.data.get_camera_xmat(cam_name).reshape(3, 3)

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot
    T_world_cam[:3, 3] = cam_pos

    # 机器人基座在世界系下的位姿
    base_pos = sim.data.get_body_xpos(base_name)
    base_rot = sim.data.get_body_xmat(base_name).reshape(3, 3)

    T_world_base = np.eye(4)
    T_world_base[:3, :3] = base_rot
    T_world_base[:3, 3] = base_pos

    # 相机 → 机器人基座
    T_cam_to_base = np.linalg.inv(T_world_cam) @ T_world_base

    print("\n=== 外参矩阵计算结果 ===")
    print(f"T_world→camera:\n{T_world_cam}")
    print(f"\nT_world→robot_base:\n{T_world_base}")
    print(f"\nT_camera→robot_base (手眼标定结果):\n{T_cam_to_base}")

    return T_cam_to_base
# =============== 新增：打印关节到 joint limit 的 margin ===============
def print_joint_limit_margins(sim, robot=None, dof=7):
    """
    打印前 dof 个关节当前角度、上下限、margin（离上下限的最小距离）。
    用来判断是不是已经快撞关节极限了。
    """
    if robot is not None and hasattr(robot, "arm_joint_names"):
        joint_names = list(getattr(robot, "arm_joint_names"))
    elif robot is not None and hasattr(robot, "joint_names"):
        joint_names = list(getattr(robot, "joint_names"))
    else:
        # 尝试通过名字猜 Sawyer 关节
        joint_names = [n for n in sim.model.joint_names if "right_j" in n][:dof]
        if not joint_names:
            joint_names = list(sim.model.joint_names[:dof])

    print("=== joint limit margins ===")
    for name in joint_names[:dof]:
        try:
            j_id = sim.model.joint_name2id(name)
        except Exception:
            continue
        qpos_adr = sim.model.jnt_qposadr[j_id]
        q = float(sim.data.qpos[qpos_adr])
        low, high = sim.model.jnt_range[j_id]
        margin = min(q - low, high - q)
        print(f"{name}: q={q:.4f}, range=({low:.4f}, {high:.4f}), margin={margin:.4f}")
# 调用计算外参矩阵
sim = env.env.sim

# =============== 新增：打印 controller 的关键信息 ===============
robot = env.env.robots[0]
ctrl = getattr(robot, "controller", getattr(robot, "_controller", None))
print("\n=== Controller 信息 ===")
for k in [
    "name",
    "input_type",
    "input_ref_frame",
    "output_max",
    "output_min",
    "control_delta",
    "position_limits",
    "orientation_limits",
]:
    print(f"{k}: {getattr(ctrl, k, None)}")

print("\n=== 初始关节到上下限的 margin ===")
print_joint_limit_margins(sim, robot=robot, dof=7)
# ================================================================


# ========== 3) 初始化 GLFW 键盘控制 ==========
if not glfw.init():
    raise Exception("无法初始化 GLFW（可能没有图形界面）")

# 创建一个隐藏的键盘监听窗口
glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
keyboard_window = glfw.create_window(200, 100, "Keyboard Control", None, None)
glfw.make_context_current(keyboard_window)

# action 含义：[dx, dy, dz, dRx, dRy, dRz, gripper]
action = np.zeros(7)

step_size = 1
step_size1 = 0.5
rot_step = 0.05

print("\n=== 实时控制说明 ===")
print("W/S: 前后 | A/D: 左右 | R/F: 上下")
print("J/L/I/K/U/O: 旋转 | Z/X: 合拢/张开 | Q: 退出\n")
step_idx = 0
prev = None

def almost_stuck(curr, prev, eps=1e-4):
    if prev is None: return False
    return np.linalg.norm(curr - prev) < eps

# 尝试从 controller 里取“目标/goal”（不同版本字段名不同，所以多试几个）
def get_goal(ctrl):
    for k in ["goal_pos", "desired_pos", "target_pos", "goal", "desired_pose", "target_pose"]:
        if hasattr(ctrl, k):
            v = getattr(ctrl, k)
            try:
                v = np.asarray(v).squeeze()
                return k, v
            except Exception:
                pass
    return None, None

# ========== 4) 控制循环 ==========
try:
    while not glfw.window_should_close(keyboard_window):
        glfw.poll_events()
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
        # =============== 新增：T 键做一次 0 动作漂移测试 ===============
        if glfw.get_key(keyboard_window, glfw.KEY_T) == glfw.PRESS:
            print("\n=== Zero-action 测试（60 步）=== ")
            zero = np.zeros(7)
            for i in range(60):
                obs, r, d, info = env.step(zero)
                eef_pos = _extract_eef_pos(obs, sim)
                if i in [0, 1, 2, 10, 59]:
                    print(f"[zero {i:02d}] EEF pos: {eef_pos}")
                env.env.render()
            # 测试后把 action 清零并保持夹爪默认
            action = np.zeros(7)
            action[-1] = -1.0
            # 跳过本轮其它逻辑
            continue

        # 退出
        if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
            print("退出控制循环。")
            break

        obs, reward, done, info = env.step(action)
        if sim.data.ncon > 0:
            print(">>> CONTACT ncon =", sim.data.ncon)
            for i in range(min(sim.data.ncon, 6)):
                c = sim.data.contact[i]
                print(sim.model.geom_id2name(c.geom1), "<->", sim.model.geom_id2name(c.geom2))

        if not np.array_equal(action, action_copy):
            step_idx += 1
            eef_pos = _extract_eef_pos(obs, sim)
            arm_qpos = _extract_arm_qpos(obs, sim)
            print(f"[step {step_idx:06d}] EEF pos: {eef_pos} | arm q(rad): {arm_qpos}")
            # 新增：每次你真正发出动作时，打印一次关节 margin
            print_joint_limit_margins(sim, robot=robot, dof=7)

        env.env.render()
        action = np.zeros(7)
        action[-1] = -1.0

finally:
    env.close()
    glfw.destroy_window(keyboard_window)
    glfw.terminate()
    print("仿真结束。")
