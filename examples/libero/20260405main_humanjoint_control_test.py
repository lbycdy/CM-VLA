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

# ========== 3) 获取机械臂关节信息 ==========
sim = env.env.sim
print(f"控制器类型: {env.env.robots[0].controller_config['type']}")

# 获取机械臂关节索引
arm_joint_indices = []
for i in range(sim.model.nu):
    actuator_name = sim.model.actuator_names[i]
    if 'robot0' in actuator_name and 'gripper' not in actuator_name:
        arm_joint_indices.append(i)
        print(f"关节 {i}: {actuator_name}")

print(f"机械臂关节索引: {arm_joint_indices}")

# 获取初始关节位置
initial_joint_pos = sim.data.qpos[arm_joint_indices].copy()
print(f"初始关节位置: {initial_joint_pos}")

# 获取关节范围
joint_limits = []
for idx in arm_joint_indices:
    joint_id = sim.model.actuator_trnid[idx, 0]
    if joint_id < sim.model.jnt_range.shape[0]:
        joint_limits.append(sim.model.jnt_range[joint_id])
    else:
        joint_limits.append([-np.pi, np.pi])  # 默认范围

print(f"关节限制: {joint_limits}")
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

# ========== 4) 初始化 GLFW 键盘控制 ==========
if not glfw.init():
    raise Exception("无法初始化 GLFW（可能没有图形界面）")

glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
keyboard_window = glfw.create_window(300, 100, "Direct Joint Control", None, None)
glfw.make_context_current(keyboard_window)

# 关节控制参数
joint_step = 0.03  # 关节角度变化步长
current_joint_pos = initial_joint_pos.copy()

print("\n=== 关节控制说明 ===")
print("数字键1-7: 增加对应关节角度")
print("Shift+1-7: 减少对应关节角度 (按住Shift再按数字键)")
print("Z/X: 夹爪合拢/张开")
print("R: 重置所有关节到初始位置")
print("P: 打印当前关节信息")
print("Q: 退出\n")


# ========== 5) 直接关节控制循环 ==========
try:
    while not glfw.window_should_close(keyboard_window):
        glfw.poll_events()

        joint_updated = False

        # 获取Shift键状态
        shift_pressed = (glfw.get_key(keyboard_window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                         glfw.get_key(keyboard_window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # 处理数字键1-7
        for i in range(min(len(arm_joint_indices), 7)):
            key = getattr(glfw, f'KEY_{i + 1}')
            if glfw.get_key(keyboard_window, key) == glfw.PRESS:
                if shift_pressed:
                    # Shift + 数字键 = 减少关节角度
                    current_joint_pos[i] -= joint_step
                    print(f"关节{i + 1}角度减少: {current_joint_pos[i]:.3f} rad")
                else:
                    # 仅数字键 = 增加关节角度
                    current_joint_pos[i] += joint_step
                    print(f"关节{i + 1}角度增加: {current_joint_pos[i]:.3f} rad")
                joint_updated = True
                # 新增：每次你真正发出动作时，打印一次关节 margin
                print_joint_limit_margins(sim, robot=robot, dof=7)

        # 应用关节限制
        for i in range(len(current_joint_pos)):
            current_joint_pos[i] = np.clip(current_joint_pos[i],
                                           joint_limits[i][0],
                                           joint_limits[i][1])

        # 夹爪控制（仍然使用动作）
        action = np.zeros(7)  # 7维动作，最后一位是夹爪
        if glfw.get_key(keyboard_window, glfw.KEY_Z) == glfw.PRESS:
            action[6] = 1.0  # 关闭夹爪
            joint_updated = True
        if glfw.get_key(keyboard_window, glfw.KEY_X) == glfw.PRESS:
            action[6] = -1.0  # 打开夹爪
            joint_updated = True

        # 重置关节位置
        if glfw.get_key(keyboard_window, glfw.KEY_R) == glfw.PRESS:
            current_joint_pos = initial_joint_pos.copy()
            action[6] = -1.0  # 打开夹爪
            joint_updated = True
            print("重置到初始关节位置")

        # 打印关节信息
        if glfw.get_key(keyboard_window, glfw.KEY_P) == glfw.PRESS:
            print(f"当前关节角度: {[f'{x:.8f}' for x in current_joint_pos]}")
            eef_pos = sim.data.get_body_xpos("robot0_right_hand")
            eef_quat = sim.data.get_body_xquat("robot0_right_hand")

            print(f"末端执行器位置: {eef_pos}")
            print(f"末端执行器角度: {eef_quat}")

        # 退出
        if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
            print("退出控制循环。")
            break

        # 如果关节位置有更新，直接设置关节位置
        if joint_updated:
            # 直接设置关节位置
            for idx, pos in zip(arm_joint_indices, current_joint_pos):
                sim.data.qpos[idx] = pos

            # 更新仿真
            sim.forward()

            # 执行一个空动作（为了渲染更新）
            obs, reward, done, info = env.step(action)

        # 渲染环境
        env.env.render()
        time.sleep(0.01)

finally:
    env.close()
    glfw.destroy_window(keyboard_window)
    glfw.terminate()
    print("仿真结束。")