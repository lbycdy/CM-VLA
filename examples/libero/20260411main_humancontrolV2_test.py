import os
import time
import numpy as np
import glfw

# 必须尽量早设置（在创建 mujoco/robosuite 环境之前）
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv


def _extract_eef_pos(obs, sim=None):
    if isinstance(obs, dict):
        for k in ("robot0_eef_pos", "eef_pos", "eef_position"):
            if k in obs:
                return np.asarray(obs[k], dtype=float).reshape(3, )
    if sim is not None:
        for site in ("gripper0_grip_site", "robot0_grip_site", "robot0_gripper_site", "eef_site"):
            try:
                return np.asarray(sim.data.get_site_xpos(site), dtype=float).reshape(3, )
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


def get_hand_eye_matrix(sim, cam_name="frontview", base_name="robot0_base"):
    """
    从仿真环境计算外参矩阵：
      T_camera→robot = inv(T_world_camera) @ T_world_robot
    """
    cam_pos = sim.data.get_camera_xpos(cam_name)
    cam_rot = sim.data.get_camera_xmat(cam_name).reshape(3, 3)

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot
    T_world_cam[:3, 3] = cam_pos

    base_pos = sim.data.get_body_xpos(base_name)
    base_rot = sim.data.get_body_xmat(base_name).reshape(3, 3)

    T_world_base = np.eye(4)
    T_world_base[:3, :3] = base_rot
    T_world_base[:3, 3] = base_pos

    T_cam_to_base = np.linalg.inv(T_world_cam) @ T_world_base

    print("\n=== 外参矩阵计算结果 ===")
    print(f"T_world→camera:\n{T_world_cam}")
    print(f"\nT_world→robot_base:\n{T_world_base}")
    print(f"\nT_camera→robot_base (手眼标定结果):\n{T_cam_to_base}")

    return T_cam_to_base


def print_joint_limit_margins(sim, robot=None, dof=7):
    """
    打印前 dof 个关节当前角度、上下限、margin（离上下限的最小距离）。
    """
    if robot is not None and hasattr(robot, "arm_joint_names"):
        joint_names = list(getattr(robot, "arm_joint_names"))
    elif robot is not None and hasattr(robot, "joint_names"):
        joint_names = list(getattr(robot, "joint_names"))
    else:
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


def main():
    # ========== 1) 初始化任务 ==========
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90"
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

    # 调用计算外参矩阵 / 打印 controller 信息
    sim = env.env.sim
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

    # ========== 3) 初始化 GLFW 键盘控制（关键修改点） ==========
    # 只创建"接收键盘事件"的窗口：不创建 OpenGL context
    if not glfw.init():
        raise RuntimeError("无法初始化 GLFW（可能没有图形界面/转发失败）")

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # ✅ 没有 OpenGL/GLX context
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    keyboard_window = glfw.create_window(200, 100, "Keyboard Control", None, None)
    if not keyboard_window:
        raise RuntimeError("Failed to create keyboard window")

    # 控制参数 - 使用更小的步长
    pos_step = 1 # 5毫米
    rot_step = 0.05  # 约0.5度

    # 夹爪状态
    gripper_state = -1.0

    # 按键状态跟踪（避免重复触发）
    key_states = {}

    # 存储动作的队列（用于平滑停止）
    action_queue = []

    print("\n=== 点击式控制说明 ===")
    print("按一下W/S/A/D/R/F：移动固定距离（5mm）")
    print("按一下J/L/I/K/U/O：旋转固定角度（0.5度）")
    print("Z/X：夹爪开合（按一次切换状态）")
    print("Q：退出")
    print("\n特点：每次按键只发送一次动作，机械臂自然停止，无回弹无抖动\n")

    step_idx = 0
    warned_render = False  # render() 失败只提示一次，避免刷屏

    try:
        while not glfw.window_should_close(keyboard_window):
            glfw.poll_events()

            # 初始化动作为零
            action = np.zeros(7)
            action[6] = gripper_state  # 保持夹爪状态

            # 按键处理：每次按下只执行一次
            # 平移控制
            if glfw.get_key(keyboard_window, glfw.KEY_W) == glfw.PRESS:
                if not key_states.get('W', False):
                    action[1] = pos_step
                    print(f"[{step_idx}] 向前移动 {pos_step:.3f}m")
                    key_states['W'] = True
            else:
                key_states['W'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_S) == glfw.PRESS:
                if not key_states.get('S', False):
                    action[1] = -pos_step
                    print(f"[{step_idx}] 向后移动 {pos_step:.3f}m")
                    key_states['S'] = True
            else:
                key_states['S'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_A) == glfw.PRESS:
                if not key_states.get('A', False):
                    action[0] = -pos_step
                    print(f"[{step_idx}] 向左移动 {pos_step:.3f}m")
                    key_states['A'] = True
            else:
                key_states['A'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_D) == glfw.PRESS:
                if not key_states.get('D', False):
                    action[0] = pos_step
                    print(f"[{step_idx}] 向右移动 {pos_step:.3f}m")
                    key_states['D'] = True
            else:
                key_states['D'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_R) == glfw.PRESS:
                if not key_states.get('R', False):
                    action[2] = pos_step
                    print(f"[{step_idx}] 向上移动 {pos_step:.3f}m")
                    key_states['R'] = True
            else:
                key_states['R'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_F) == glfw.PRESS:
                if not key_states.get('F', False):
                    action[2] = -pos_step
                    print(f"[{step_idx}] 向下移动 {pos_step:.3f}m")
                    key_states['F'] = True
            else:
                key_states['F'] = False

            # 旋转控制
            if glfw.get_key(keyboard_window, glfw.KEY_J) == glfw.PRESS:
                if not key_states.get('J', False):
                    action[3] = -rot_step
                    print(f"[{step_idx}] 绕X轴旋转 {-rot_step:.3f}rad")
                    key_states['J'] = True
            else:
                key_states['J'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_L) == glfw.PRESS:
                if not key_states.get('L', False):
                    action[3] = rot_step
                    print(f"[{step_idx}] 绕X轴旋转 {rot_step:.3f}rad")
                    key_states['L'] = True
            else:
                key_states['L'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_I) == glfw.PRESS:
                if not key_states.get('I', False):
                    action[4] = rot_step
                    print(f"[{step_idx}] 绕Y轴旋转 {rot_step:.3f}rad")
                    key_states['I'] = True
            else:
                key_states['I'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_K) == glfw.PRESS:
                if not key_states.get('K', False):
                    action[4] = -rot_step
                    print(f"[{step_idx}] 绕Y轴旋转 {-rot_step:.3f}rad")
                    key_states['K'] = True
            else:
                key_states['K'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_U) == glfw.PRESS:
                if not key_states.get('U', False):
                    action[5] = rot_step
                    print(f"[{step_idx}] 绕Z轴旋转 {rot_step:.3f}rad")
                    key_states['U'] = True
            else:
                key_states['U'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_O) == glfw.PRESS:
                if not key_states.get('O', False):
                    action[5] = -rot_step
                    print(f"[{step_idx}] 绕Z轴旋转 {-rot_step:.3f}rad")
                    key_states['O'] = True
            else:
                key_states['O'] = False

            # 夹爪控制
            if glfw.get_key(keyboard_window, glfw.KEY_Z) == glfw.PRESS:
                if not key_states.get('Z', False):
                    gripper_state = 1.0
                    action[6] = gripper_state
                    print(f"[{step_idx}] 夹爪闭合")
                    key_states['Z'] = True
            else:
                key_states['Z'] = False

            if glfw.get_key(keyboard_window, glfw.KEY_X) == glfw.PRESS:
                if not key_states.get('X', False):
                    gripper_state = -1.0
                    action[6] = gripper_state
                    print(f"[{step_idx}] 夹爪张开")
                    key_states['X'] = True
            else:
                key_states['X'] = False

            # zero-action 测试
            if glfw.get_key(keyboard_window, glfw.KEY_T) == glfw.PRESS:
                print("\n=== Zero-action 测试（60 步）=== ")
                zero = np.zeros(7)
                for i in range(60):
                    obs, r, d, info = env.step(zero)
                    eef_pos = _extract_eef_pos(obs, sim)
                    if i in [0, 1, 2, 10, 59]:
                        print(f"[zero {i:02d}] EEF pos: {eef_pos}")
                    # render 可能在 SSH X11 下失败，这里兜底
                    try:
                        env.env.render()
                    except Exception as e:
                        if not warned_render:
                            print(f"[warn] env.env.render() 失败（可忽略）：{type(e).__name__}: {e}")
                            warned_render = True
                continue

            # 退出
            if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
                print("退出控制循环。")
                break

            # 执行动作（如果有非零动作）
            if np.any(action[:6] != 0):  # 检查前6个元素是否非零
                step_idx += 1
                obs, reward, done, info = env.step(action)

                time.sleep(0.05)  # 50ms让机械臂稳定

                # 获取当前末端位置
                eef_pos = _extract_eef_pos(obs, sim)
                print(f"[{step_idx}] 末端位置: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

                # 然后立即发送一次零动作，帮助稳定
                # obs, reward, done, info = env.step(np.zeros(7))
            else:
                # 没有移动指令，只发送夹爪状态
                zero_action = np.zeros(7)
                zero_action[6] = gripper_state
                # obs, reward, done, info = env.step(zero_action)



            # render 可能在 SSH X11 下失败，这里兜底
            try:
                env.env.render()
            except Exception as e:
                if not warned_render:
                    print(f"[warn] env.env.render() 失败（可忽略）：{type(e).__name__}: {e}")
                    warned_render = True

            # 控制频率
            time.sleep(0.01)

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            if keyboard_window is not None:
                glfw.destroy_window(keyboard_window)
        finally:
            glfw.terminate()
        print("仿真结束。")


if __name__ == "__main__":
    main()