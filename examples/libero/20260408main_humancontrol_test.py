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
    task_suite_name = "libero_10"#libero_spatial，libero_object
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
    # 只创建“接收键盘事件”的窗口：不创建 OpenGL context
    if not glfw.init():
        raise RuntimeError("无法初始化 GLFW（可能没有图形界面/转发失败）")

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)   # ✅ 没有 OpenGL/GLX context
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    keyboard_window = glfw.create_window(200, 100, "Keyboard Control", None, None)
    if not keyboard_window:
        raise RuntimeError("Failed to create keyboard window")

    # ❌ 不要再调用 glfw.make_context_current(keyboard_window)
    # 因为 NO_API 窗口没有 OpenGL context，会直接报你遇到的 65546 错误

    # action 含义：[dx, dy, dz, dRx, dRy, dRz, gripper]
    action = np.zeros(7)

    step_size = 1
    rot_step = 0.05

    print("\n=== 实时控制说明 ===")
    print("W/S: 前后 | A/D: 左右 | R/F: 上下")
    print("J/L/I/K/U/O: 旋转 | Z/X: 合拢/张开 | Q: 退出")
    print("T: 做一次 zero-action 测试（60 步）\n")

    step_idx = 0
    warned_render = False  # render() 失败只提示一次，避免刷屏

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

                action = np.zeros(7)
                action[-1] = -1.0
                continue

            # 退出
            if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
                print("退出控制循环。")
                break

            obs, reward, done, info = env.step(action)

            # 打印 contact
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
                print_joint_limit_margins(sim, robot=robot, dof=7)

            # render 可能在 SSH X11 下失败，这里兜底（你有 camera_obs 的话其实不 render 也行）
            try:
                env.env.render()
            except Exception as e:
                if not warned_render:
                    print(f"[warn] env.env.render() 失败（可忽略）：{type(e).__name__}: {e}")
                    warned_render = True

            # 每步后把 action 清零并保持夹爪默认
            action = np.zeros(7)
            action[-1] = -1.0

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
