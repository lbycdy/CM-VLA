import os
import time
import numpy as np
import glfw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv
import os
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





# ========== 3) 初始化 GLFW 键盘控制 ==========
if not glfw.init():
    raise Exception("无法初始化 GLFW（可能没有图形界面）")

# 创建一个隐藏的键盘监听窗口
glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
keyboard_window = glfw.create_window(200, 100, "Keyboard Control", None, None)
glfw.make_context_current(keyboard_window)

# action 含义：[dx, dy, dz, dRx, dRy, dRz, gripper]
action = np.zeros(7)
action[-1] = -1.0
step_size = 0.01
rot_step = 0.05

print("\n=== 实时控制说明 ===")
print("W/S: 前后 | A/D: 左右 | R/F: 上下")
print("J/L/I/K/U/O: 旋转 | Z/X: 合拢/张开 | Q: 退出\n")

time1 = time.time()

# ========== 4) 控制循环 ==========
try:
    while not glfw.window_should_close(keyboard_window):
        time2 = time.time()
        # if time2 - time1> 10:
        #     mj_sim = env.env.sim
        #     # 获取所有关节位置向量（包括抓手等）
        #     qpos = mj_sim.data.qpos.copy()
        #     # 取前 7 个臂部关节角（根据你的机器人型号调整索引）
        #     arm_qpos = qpos[:7]
        #     print("7DOF 关节角:", arm_qpos)


        glfw.poll_events()
        action_copy = action.copy()
        # 平移
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

        # 旋转
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




        env.env.render()
        if not np.array_equal(action, action_copy):
            print('obs:', obs["robot0_eef_pos"])
            print('action:',action)


            # print(env.env.robots[0].controller_config)

        # time.sleep(0.02)

        if done:
            env.reset()

finally:
    env.close()
    glfw.destroy_window(keyboard_window)
    glfw.terminate()
    print("仿真结束。")
