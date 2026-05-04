import argparse
import h5py
import numpy as np
import cv2
import json
import os
import re
from pathlib import Path

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

def process_demo(demo_file, trajectory_index, video_path, use_camera_obs, use_depth, local_pkg_root, cap_index=5):
    """
    处理单个demo文件中的指定轨迹，并录制视频
    
    Args:
        demo_file: demo.hdf5文件路径
        trajectory_index: 要运行的轨迹索引
        video_path: 输出视频的路径
        use_camera_obs: 是否使用摄像头观察
        use_depth: 是否使用深度图
        local_pkg_root: libero包的本地根目录
        cap_index: 跳过前几步
    """
    # 读取 HDF5 文件
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])
    demos = list(f["data"].keys())
    problem_name = json.loads(f["data"].attrs["problem_info"])["problem_name"]
    
    # 检查轨迹索引是否有效
    if trajectory_index >= len(demos):
        print(f"[警告] 轨迹索引 {trajectory_index} 超出范围 (最大: {len(demos)-1})，跳过此任务")
        f.close()
        return False
    
    # 获取 bddl_file_name 参数
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    
    # Fix bddl_file_name path if it points to a different user's directory
    if "bddl_files/" in bddl_file_name:
        # Extract the relative path after bddl_files/
        bddl_relative = bddl_file_name.split("bddl_files/", 1)[1]
        # Get the current script's libero path
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        libero_root = os.path.dirname(current_script_dir)
        # Reconstruct the correct path
        bddl_file_name = os.path.join(libero_root, "libero", "libero", "bddl_files", bddl_relative)
        print(f"[info] Corrected bddl path to: {bddl_file_name}")
    
    env_kwargs["bddl_file_name"] = bddl_file_name

    # 启用渲染器和摄像头观察
    env_kwargs["has_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    
    # 使用libero_utils.update_env_kwargs来更新环境参数，确保一致性
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=True,
        has_offscreen_renderer=use_camera_obs,
        ignore_done=True,
        use_camera_obs=use_camera_obs,
        camera_depths=use_depth,
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

    # 初始化视频写入器
    video_writer = None
    
    try:
        ep = demos[trajectory_index]
        # print(f"播放演示 {ep} (轨迹索引: {trajectory_index})...")

        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        
        # 使用libero_utils.postprocess_model_xml来预处理XML
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

        done = False

        for j, action in enumerate(actions):
            if done:
                print(f"Episode {ep} 已终止，重置环境")
                env.reset()
                done = False
                continue

            obs, reward, done, info = env.step(action)

            # 跳过前 cap_index 步
            if j < cap_index:
                continue

            # 获取图像用于录制视频
            if use_camera_obs:
                # 反转图像: [::-1, ::-1] (垂直和水平翻转)
                # agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agentview_image = np.ascontiguousarray(obs["agentview_image"][::, ::-1])

                # 转换为BGR用于OpenCV显示和保存
                agentview_image = cv2.cvtColor(agentview_image, cv2.COLOR_RGB2BGR)
                
                # 初始化视频写入器
                if video_writer is None:
                    height, width = agentview_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                    # frame_pause_sec = 0.5
                    # fps = (1.0 / frame_pause_sec) if (frame_pause_sec is not None and frame_pause_sec > 0) else 20.0
                    # video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # 写入帧
                video_writer.write(agentview_image)
                
                # 显示图像
                cv2.imshow("agentview", agentview_image)
                if use_depth:
                    # depth_image = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
                    depth_image = np.ascontiguousarray(obs["agentview_depth"][::, ::-1])
                    # depth是单通道，转换为3通道用于显示
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR) if len(depth_image.shape) == 2 else depth_image
                    cv2.imshow("depth", depth_image)

                cv2.waitKey(3)
                # delay_ms = int(frame_pause_sec * 1000) if (frame_pause_sec is not None and frame_pause_sec > 0) else 1
                # cv2.waitKey(max(1, delay_ms))
            else:
                # 如果不使用摄像头观察，调用 `env.render()` 来强制渲染
                env.render()
                
                # 获取渲染的图像用于录制视频
                # 注意: 这里需要从env获取渲染的图像
                # 如果env.render()不返回图像，可能需要使用其他方法

        # print(f"演示 {ep} 完成。")
        
        # 释放视频写入器
        if video_writer is not None:
            video_writer.release()
            print(f"[info] 视频已保存")
        
    finally:
        # 释放资源
        f.close()
        
        try:
            env.close()
        except Exception as e:
            print(f"关闭环境时发生错误: {e}")
        
        cv2.destroyAllWindows()
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/lbycdy/work/camaction/demonstration_data",
                       help="demonstration_data的根目录")
    parser.add_argument("--folder", default="libero_object", 
                       choices=["libero_10", "libero_goal", "libero_object"],
                       help="选择要运行的文件夹")
    parser.add_argument("--trajectory-index", type=int, default=1,
                       help="选择运行第几个轨迹 (从0开始)")
    parser.add_argument("--output-dir", default="/home/lbycdy/work/camaction",
                       help="视频输出目录")
    parser.add_argument("--use-camera-obs", action="store_true",
                       help="使用摄像头观察")
    parser.add_argument("--use-depth", action="store_true",
                       help="使用深度图")
    parser.add_argument("--cap-index", type=int, default=5,
                       help="跳过前几步")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建已完成任务记录文件
    completed_tasks_file = os.path.join(args.output_dir, f"completed_tasks_{args.folder}.txt")
    
    # 读取已完成的任务
    completed_tasks = set()
    if os.path.exists(completed_tasks_file):
        with open(completed_tasks_file, 'r') as f:
            completed_tasks = set(line.strip() for line in f)
    
    # 获取本地libero包根目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    libero_root = os.path.dirname(current_script_dir)
    local_pkg_root = os.path.join(libero_root, "libero", "libero")
    
    # 获取选定文件夹的路径
    folder_path = os.path.join(args.data_root, args.folder)
    
    if not os.path.exists(folder_path):
        print(f"[错误] 文件夹不存在: {folder_path}")
        return
    
    # 获取所有任务文件夹
    task_folders = sorted([d for d in os.listdir(folder_path) 
                          if os.path.isdir(os.path.join(folder_path, d))])
    
    print(f"[info] 在 {args.folder} 中找到 {len(task_folders)} 个任务")
    print(f"[info] 已完成 {len(completed_tasks)} 个任务")
    # print(f"[info] 将运行轨迹索引: {args.trajectory_index}")
    
    # 按顺序处理每个任务
    for idx, task_folder in enumerate(task_folders):
        task_name = task_folder
        
        # 检查任务是否已完成
        if task_name in completed_tasks:
            print(f"\n[{idx+1}/{len(task_folders)}] 任务 {task_name} 已完成，跳过")
            continue
        
        print(f"\n[{idx+1}/{len(task_folders)}] 处理任务: {task_name}")
        
        # 构建demo文件路径
        demo_file = os.path.join(folder_path, task_folder, "demo.hdf5")
        
        if not os.path.exists(demo_file):
            print(f"[警告] demo文件不存在: {demo_file}，跳过")
            continue
        
        # 构建视频输出路径
        video_filename = f"{task_name}_traj{args.trajectory_index}.mp4"
        video_path = os.path.join(args.output_dir, args.folder, video_filename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # 处理demo
        try:
            success = process_demo(
                demo_file=demo_file,
                trajectory_index=args.trajectory_index,
                video_path=video_path,
                use_camera_obs=args.use_camera_obs,
                use_depth=args.use_depth,
                local_pkg_root=local_pkg_root,
                cap_index=args.cap_index
            )
            
            if success:
                # 记录已完成的任务
                with open(completed_tasks_file, 'a') as f:
                    f.write(f"{task_name}\n")
                completed_tasks.add(task_name)
                print(f"[成功] 视频已保存到: {video_path}")
            
        except Exception as e:
            print(f"[错误] 处理任务 {task_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[完成] 所有任务处理完毕！")
    print(f"[完成] 共完成 {len(completed_tasks)} 个任务")

if __name__ == "__main__":
    main()
