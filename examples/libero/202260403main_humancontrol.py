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
    "use_camera_obs": False,
    "has_offscreen_renderer": False,
    "render_camera": "frontview",
}
env = OnScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

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

# 调用计算外参矩阵
sim = env.env.sim


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
        if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
            print("退出控制循环。")
            break

        # # 执行动作
        # if not np.array_equal(action, action_copy):
        #     action[3] += step_size1
        #     action[4] += step_size1
        #     action[5] += step_size1
        obs, reward, done, info = env.step(action)
        #     print('EEF position:', obs["robot0_eef_pos"])
        #     print("EEF orientation (quat):", obs["robot0_eef_quat"])
        #     print('action:', action)
        #     get_hand_eye_matrix(sim, cam_name="frontview", base_name="robot0_base")
        #
        #     if done:
        #         env.reset()
        env.env.render()
        action = np.zeros(7)
        action[-1] = -1.0

finally:
    env.close()
    glfw.destroy_window(keyboard_window)
    glfw.terminate()
    print("仿真结束。")
