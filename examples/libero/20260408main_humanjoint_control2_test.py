import os
import time
import numpy as np
import glfw

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OnScreenRenderEnv

# ✅ 尽量早设置（创建 env 前）
os.environ.setdefault("MUJOCO_GL", "egl")


def _pad_or_truncate_flat_state(flat_state: np.ndarray, sim):
    """
    LIBERO 给的 init_state 有时长度和当前 sim 不一致（常见是少了 qvel / act 部分）。
    这里按 MuJoCo 期望的 (nq + nv + na) 自动补零或截断，避免 set_state_from_flattened 报错。
    """
    flat = np.asarray(flat_state, dtype=np.float64).reshape(-1)
    nq = int(sim.model.nq)
    nv = int(sim.model.nv)
    na = int(getattr(sim.model, "na", 0))
    expected = nq + nv + na

    if flat.size != expected:
        print(
            f"[warn] init_state length mismatch: got={flat.size}, expected={expected} "
            f"(nq={nq}, nv={nv}, na={na}). Will pad/truncate with zeros."
        )
        if flat.size < expected:
            flat = np.concatenate([flat, np.zeros(expected - flat.size, dtype=flat.dtype)], axis=0)
        else:
            flat = flat[:expected]
    return flat


def _safe_set_init_state(env, init_state):
    """
    尝试 set_init_state（会生成 obs）；若失败则不崩溃，继续用 reset 的默认初始状态。
    """
    sim = env.env.sim
    flat = _pad_or_truncate_flat_state(init_state, sim)
    try:
        env.set_init_state(flat)
        sim.forward()
        print("[info] set_init_state OK")
        return True
    except Exception as e:
        print(f"[warn] set_init_state failed, fallback to default reset state: {type(e).__name__}: {e}")
        try:
            sim.forward()
        except Exception:
            pass
        return False


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

# ✅ 先拿到 sim，再做 init_state pad/truncate
sim = env.env.sim
print(f"[info] sim sizes: nq={sim.model.nq}, nv={sim.model.nv}, na={getattr(sim.model, 'na', 0)}")

# ✅ 这里改：安全设置 init_state（不会因为长度不匹配直接崩）
init_states = task_suite.get_task_init_states(task_id)
_safe_set_init_state(env, init_states[0])

# ========== 3) 获取机械臂关节信息 ==========
robot = env.env.robots[0]
print(f"控制器类型: {robot.controller_config['type']}")


def _joint_name_by_id(model, jid: int):
    try:
        return model.joint_id2name(jid)  # mujoco_py 风格
    except Exception:
        try:
            return model.joint_names[jid]
        except Exception:
            return f"joint_{jid}"


# ✅ actuator -> joint_id -> qpos_adr / dof_adr
arm_actuator_ids = []
arm_joint_ids = []
arm_qpos_addrs = []
arm_qvel_addrs = []
joint_limits = []

for act_id in range(sim.model.nu):
    actuator_name = sim.model.actuator_names[act_id]
    if ("robot0" in actuator_name) and ("gripper" not in actuator_name):
        joint_id = int(sim.model.actuator_trnid[act_id, 0])
        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
        dof_adr = int(sim.model.jnt_dofadr[joint_id])

        if joint_id < sim.model.jnt_range.shape[0]:
            j_low, j_high = sim.model.jnt_range[joint_id]
        else:
            j_low, j_high = -np.pi, np.pi

        arm_actuator_ids.append(act_id)
        arm_joint_ids.append(joint_id)
        arm_qpos_addrs.append(qpos_adr)
        arm_qvel_addrs.append(dof_adr)
        joint_limits.append((float(j_low), float(j_high)))

        print(
            f"关节 {len(arm_joint_ids)}: actuator_id={act_id} '{actuator_name}' -> "
            f"joint_id={joint_id} '{_joint_name_by_id(sim.model, joint_id)}' "
            f"(qpos_adr={qpos_adr}, dof_adr={dof_adr})"
        )

print(f"机械臂 actuator_ids: {arm_actuator_ids}")
print(f"机械臂 joint_ids   : {arm_joint_ids}")
print(f"机械臂 qpos_addrs  : {arm_qpos_addrs}")

initial_joint_pos = sim.data.qpos[arm_qpos_addrs].copy()
print(f"初始关节位置: {initial_joint_pos}")
print(f"关节限制: {joint_limits}")


def print_joint_limit_margins(sim, robot=None, dof=6):
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
print_joint_limit_margins(sim, robot=robot, dof=6)

# ========== 4) 初始化 GLFW 键盘控制（NO_API，不建 OpenGL context） ==========
if not glfw.init():
    raise RuntimeError("无法初始化 GLFW（可能没有图形界面 / X11 转发不可用）")

glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.VISIBLE, glfw.TRUE)

keyboard_window = glfw.create_window(300, 100, "Direct Joint Control", None, None)
if not keyboard_window:
    raise RuntimeError("Failed to create keyboard window")

joint_step = 0.03
current_joint_pos = initial_joint_pos.copy()

print("\n=== 关节控制说明 ===")
print("数字键1-6: 增加对应关节角度")
print("Shift+1-6: 减少对应关节角度 (按住Shift再按数字键)")
print("Z/X: 夹爪合拢/张开")
print("R: 重置所有关节到初始位置")
print("P: 打印当前关节信息")
print("Q: 退出\n")

warned_render = False

# ========== 5) 控制循环 ==========
try:
    while not glfw.window_should_close(keyboard_window):
        glfw.poll_events()
        joint_updated = False

        shift_pressed = (
            glfw.get_key(keyboard_window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(keyboard_window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        n_ctrl = min(len(current_joint_pos), 6)
        for i in range(n_ctrl):
            key = getattr(glfw, f"KEY_{i + 1}")
            if glfw.get_key(keyboard_window, key) == glfw.PRESS:
                if shift_pressed:
                    current_joint_pos[i] -= joint_step
                    print(f"关节{i + 1}角度减少: {current_joint_pos[i]:.3f} rad")
                else:
                    current_joint_pos[i] += joint_step
                    print(f"关节{i + 1}角度增加: {current_joint_pos[i]:.3f} rad")
                joint_updated = True
                print_joint_limit_margins(sim, robot=robot, dof=6)

        for i in range(len(current_joint_pos)):
            low_i, high_i = joint_limits[i]
            current_joint_pos[i] = np.clip(current_joint_pos[i], low_i, high_i)

        action = np.zeros(7, dtype=float)
        if glfw.get_key(keyboard_window, glfw.KEY_Z) == glfw.PRESS:
            action[6] = 1.0
            joint_updated = True
        if glfw.get_key(keyboard_window, glfw.KEY_X) == glfw.PRESS:
            action[6] = -1.0
            joint_updated = True

        if glfw.get_key(keyboard_window, glfw.KEY_R) == glfw.PRESS:
            current_joint_pos = initial_joint_pos.copy()
            action[6] = -1.0
            joint_updated = True
            print("重置到初始关节位置")

        if glfw.get_key(keyboard_window, glfw.KEY_P) == glfw.PRESS:
            print(f"当前关节角度: {[f'{x:.8f}' for x in current_joint_pos]}")
            try:
                eef_pos = sim.data.get_body_xpos("robot0_right_hand")
                eef_quat = sim.data.get_body_xquat("robot0_right_hand")
                print(f"末端执行器位置: {eef_pos}")
                print(f"末端执行器角度: {eef_quat}")
            except Exception as e:
                print("[warn] 末端执行器信息读取失败：", e)

        if glfw.get_key(keyboard_window, glfw.KEY_Q) == glfw.PRESS:
            print("退出控制循环。")
            break

        if joint_updated:
            for qpos_adr, qvel_adr, pos in zip(arm_qpos_addrs, arm_qvel_addrs, current_joint_pos):
                sim.data.qpos[qpos_adr] = pos
                try:
                    sim.data.qvel[qvel_adr] = 0.0
                except Exception:
                    pass

            sim.forward()
            _obs, _reward, _done, _info = env.step(action)

        try:
            env.env.render()
        except Exception as e:
            if not warned_render:
                print(f"[warn] env.env.render() 失败（可忽略）：{type(e).__name__}: {e}")
                warned_render = True

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
