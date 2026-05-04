import argparse
import h5py
import numpy as np
import cv2
import json
import os
import re

from libero.libero.utils import utils as libero_utils
from libero.libero.envs import *

def remap_mujoco_asset_paths(xml: str, local_libero_pkg_root: str) -> str:
    """
    把 XML 中所有形如 file=".../assets/xxx" 的路径，替换为本机的:
      <local_libero_pkg_root>/assets/xxx

    local_libero_pkg_root 例子：
      /home/new1/server/libero/libero/libero
    """
    local_assets_root = os.path.join(local_libero_pkg_root, "assets")

    def _replace(m):
        path = m.group(1)
        # 只处理带 /assets/ 的路径（mesh/texture/include 常见）
        if "/assets/" not in path:
            return m.group(0)

        suffix = path.split("/assets/", 1)[1]  # assets/ 后面的相对部分
        new_path = os.path.join(local_assets_root, suffix)

        # 只有当旧路径不存在、新路径存在时才替换（更安全）
        if os.path.isabs(path) and (not os.path.exists(path)) and os.path.exists(new_path):
            return f'file="{new_path}"'

        # 旧路径虽然存在或新路径不存在：不动
        return m.group(0)

    # 替换所有 file="..."
    return re.sub(r'file="([^"]+)"', _replace, xml)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="/home/lbycdy/demonstration_data/libero_object/robosuite_ln_libero_floor_manipulation_1766634005_7861943_pick_the_ketchup_and_place_it_in_the_basket/demo.hdf5")
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    args = parser.parse_args()

    # 读取 HDF5 文件
    f = h5py.File(args.demo_file, "r")
    env_name = f["data"].attrs["env"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])
    demos = list(f["data"].keys())
    problem_name = json.loads(f["data"].attrs["problem_info"])["problem_name"]
    
    # 获取 bddl_file_name 参数
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    
    # Fix bddl_file_name path if it points to a different user's directory
    if "bddl_files/" in bddl_file_name:
        # Extract the relative path after bddl_files/
        bddl_relative = bddl_file_name.split("bddl_files/", 1)[1]
        # Get the current script's libero path
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        libero_root = os.path.dirname(current_script_dir)  # Go up from current script directory
        # Reconstruct the correct path
        bddl_file_name = os.path.join(libero_root, "libero", "libero", "bddl_files", bddl_relative)
        print(f"[info] Corrected bddl path to: {bddl_file_name}")
    
    env_kwargs["bddl_file_name"] = bddl_file_name  # 将 bddl_file_name 添加到 env_kwargs

    # 启用渲染器和摄像头观察
    env_kwargs["has_renderer"] = True  # 启用渲染器
    env_kwargs["use_camera_obs"] = True  # 启用摄像头观察
    
    # 使用libero_utils.update_env_kwargs来更新环境参数，确保一致性
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=True,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=["agentview"],
        reward_shaping=True,
        control_freq=20,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations=None,
    )
    
    print(f"[info] 使用bddl文件: {bddl_file_name}")
    print(f"[info] bddl文件存在: {os.path.exists(bddl_file_name)}")

    # 创建环境
    env = TASK_MAPPING[problem_name](**env_kwargs)

    cap_index = 5  # 跳过前 cap_index 步
    
    # 获取本地libero包根目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    libero_root = os.path.dirname(current_script_dir)  # Go up from current script directory
    local_pkg_root = os.path.join(libero_root, "libero", "libero")

    for ep in demos:
        print(f"播放演示 {ep}... (按 ESC 键退出)")

        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        
        # 使用libero_utils.postprocess_model_xml来预处理XML（如代码2所示）
        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        
        # 应用资产路径重映射
        model_xml = remap_mujoco_asset_paths(model_xml, local_pkg_root)
        
        # 重置环境
        env.reset_from_xml_string(model_xml)
        
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])
        
        num_actions = actions.shape[0]
        init_idx = 0
        env.sim.reset()
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()

        done = False  # 初始化 done 变量

        for j, action in enumerate(actions):
            # 如果 episode 已经终止，重置环境并跳过当前步骤
            if done:
                print(f"Episode {ep} 已终止，重置环境")
                env.reset()  # 重置环境
                done = False  # 重新初始化 done
                continue  # 跳过当前步骤

            obs, reward, done, info = env.step(action)

            # 跳过前 cap_index 步
            if j < cap_index:
                continue

            if args.use_camera_obs:
                if args.use_depth:
                    depth_image = obs["agentview_depth"]
                agentview_image = obs["agentview_image"]
                agentview_image = cv2.cvtColor(agentview_image, cv2.COLOR_RGB2BGR)

                # 显示 agentview 图像
                cv2.imshow("agentview", agentview_image)
                if args.use_depth:
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("depth", depth_image)

                # 强制更新窗口，确保图像显示
                cv2.waitKey(1)  # 持续刷新窗口

            else:
                # 如果不使用摄像头观察，调用 `env.render()` 来强制渲染
                env.render()

        print(f"演示 {ep} 完成。")

    f.close()

    # 捕获异常并在异常时关闭环境
    try:
        env.close()
    except Exception as e:
        print(f"关闭环境时发生错误: {e}")
        # 在这里处理资源释放错误，防止程序中断

    cv2.destroyAllWindows()  # 确保关闭所有 OpenCV 窗口

if __name__ == "__main__":
    main()