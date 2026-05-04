import argparse
import h5py
import numpy as np
import cv2
import json
import os
import re
import time
import gc

from libero.libero.utils import utils as libero_utils
from libero.libero.envs import *


def remap_mujoco_asset_paths(xml: str, local_libero_pkg_root: str) -> str:
    """Remap MuJoCo asset paths in XML to the local LIBERO assets directory."""
    local_assets_root = os.path.join(local_libero_pkg_root, "assets")

    def _replace(m):
        path = m.group(1)
        if "/assets/" not in path:
            return m.group(0)

        suffix = path.split("/assets/", 1)[1]
        new_path = os.path.join(local_assets_root, suffix)

        if os.path.isabs(path) and (not os.path.exists(path)) and os.path.exists(new_path):
            return f'file="{new_path}"'
        return m.group(0)

    return re.sub(r'file="([^"]+)"', _replace, xml)


def _video_seems_ok(video_path: str, min_bytes: int = 2048) -> bool:
    try:
        return os.path.exists(video_path) and os.path.getsize(video_path) >= min_bytes
    except OSError:
        return False


def process_demo(
    demo_file: str,
    trajectory_index: int,
    video_path: str,
    use_camera_obs: bool,
    use_depth: bool,
    local_pkg_root: str,
    cap_index: int = 5,
    action_pause: float = 0.5,
    video_fps: float = 20.0,
    no_display: bool = False,
    mirror_lr: bool = True,
) -> bool:
    """Replay one trajectory from a LIBERO demo and record a video.

    Important behavior:
      - Wait `action_pause` seconds between actions (real blocking wait).
      - The exported video *includes* this pause by duplicating the current frame
        for `round(video_fps * action_pause)` frames.
      - Default is to use camera obs for recording.
    """

    f = h5py.File(demo_file, "r")
    env_kwargs = json.loads(f["data"].attrs["env_info"])
    demos = list(f["data"].keys())
    problem_name = json.loads(f["data"].attrs["problem_info"])["problem_name"]

    if trajectory_index >= len(demos):
        print(f"[警告] 轨迹索引 {trajectory_index} 超出范围 (最大: {len(demos) - 1})，跳过此任务")
        f.close()
        return False

    # Resolve bddl path
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    if "bddl_files/" in bddl_file_name:
        bddl_relative = bddl_file_name.split("bddl_files/", 1)[1]
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        libero_root = os.path.dirname(current_script_dir)
        bddl_file_name = os.path.join(libero_root, "libero", "libero", "bddl_files", bddl_relative)
        print(f"[info] Corrected bddl path to: {bddl_file_name}")

    env_kwargs["bddl_file_name"] = bddl_file_name

    # Renderer settings:
    # - Recording via camera obs only needs offscreen renderer.
    # - Turning on on-screen MuJoCo viewer (has_renderer=True) is a common source of
    #   GLFW/GL teardown crashes. We keep it off by default.
    has_offscreen_renderer = bool(use_camera_obs)
    has_renderer = (not use_camera_obs) and (not no_display)

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
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

    env = TASK_MAPPING[problem_name](**env_kwargs)
    video_writer = None
    wrote_any_frame = False

    try:
        ep = demos[trajectory_index]
        model_xml = f[f"data/{ep}"].attrs["model_file"]
        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        model_xml = remap_mujoco_asset_paths(model_xml, local_pkg_root)

        env.reset_from_xml_string(model_xml)

        states = f[f"data/{ep}/states"][()]
        actions = np.array(f[f"data/{ep}/actions"][()])
        print(f"[info] actions shape: {actions.shape}")

        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        done = False
        for j, action in enumerate(actions):
            if done:
                env.reset()
                done = False
                continue

            obs, reward, done, info = env.step(action)

            # Optional early skip (still waits, but doesn't write frames)
            if j < cap_index:
                if action_pause and action_pause > 0:
                    if no_display:
                        time.sleep(float(action_pause))
                    else:
                        cv2.waitKey(int(round(float(action_pause) * 1000)))
                continue

            if not use_camera_obs:
                # No camera obs => cannot record video in this script.
                if action_pause and action_pause > 0:
                    time.sleep(float(action_pause))
                continue

            if "agentview_image" not in obs:
                raise KeyError("obs does not contain 'agentview_image' (check camera settings)")

            img = obs["agentview_image"]
            if mirror_lr:
                img = img[::-1, ::-1]
            img = np.ascontiguousarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, float(video_fps), (w, h))

            hold_frames = 1
            if action_pause and action_pause > 0:
                hold_frames = max(1, int(round(float(video_fps) * float(action_pause))))
            for _ in range(hold_frames):
                video_writer.write(img)
                wrote_any_frame = True

            if not no_display:
                cv2.imshow("agentview", img)
                if use_depth and ("agentview_depth" in obs):
                    depth = obs["agentview_depth"]
                    if mirror_lr:
                        depth = depth[::-1, ::-1]
                    depth = np.ascontiguousarray(depth)
                    if len(depth.shape) == 2:
                        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
                    cv2.imshow("depth", depth)
                delay_ms = int(round(float(action_pause) * 1000)) if (action_pause and action_pause > 0) else 1
                key = cv2.waitKey(max(1, delay_ms)) & 0xFF
                if key == ord("q"):
                    break
            else:
                if action_pause and action_pause > 0:
                    time.sleep(float(action_pause))

        if video_writer is not None:
            video_writer.release()

        if not wrote_any_frame:
            print("[警告] 没有写入任何视频帧：可能 cap-index 太大、轨迹太短、或未启用 camera obs")
            return False

        if _video_seems_ok(video_path):
            print("[info] 视频已保存")
            return True
        else:
            print("[警告] 视频文件看起来没有成功写出（文件不存在或过小）")
            return False

    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        try:
            env.close()
        except Exception as e:
            print(f"关闭环境时发生错误: {e}")
        if not no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        # Help some GL backends clean up deterministically
        try:
            del env
        except Exception:
            pass
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/lbycdy/work/camaction/demonstration_data",
                        help="demonstration_data的根目录")
    parser.add_argument("--folder", default="libero_10",
                        choices=["libero_10", "libero_goal", "libero_object"],
                        help="选择要运行的文件夹")
    parser.add_argument("--trajectory-index", type=int, default=1,
                        help="选择运行第几个轨迹 (从0开始)")
    parser.add_argument("--output-dir", default="/home/lbycdy/work/camaction",
                        help="视频输出目录")

    cam_group = parser.add_mutually_exclusive_group()
    cam_group.add_argument("--use-camera-obs", dest="use_camera_obs", action="store_true",
                           help="使用摄像头观察/录制（默认开启）")
    cam_group.add_argument("--no-camera-obs", dest="use_camera_obs", action="store_false",
                           help="禁用摄像头观察/录制")
    parser.set_defaults(use_camera_obs=True)

    parser.add_argument("--use-depth", action="store_true", help="使用深度图")
    parser.add_argument("--cap-index", type=int, default=5, help="跳过前几步")
    parser.add_argument("--action-pause", type=float, default=0.0,
                        help="每个action执行后等待的秒数（视频也会包含这个停顿）")
    parser.add_argument("--video-fps", type=float, default=600.0,
                        help="输出视频帧率（停顿通过重复写帧体现）")
    parser.add_argument("--no-display", action="store_true", help="不显示OpenCV窗口（仅保存视频）")
    parser.add_argument("--no-mirror-lr", dest="mirror_lr", action="store_false",
                        help="关闭左右镜像（默认开启）")
    parser.set_defaults(mirror_lr=True)

    parser.add_argument("--force", action="store_true",
                        help="忽略 completed_tasks 文件，强制重新处理所有任务")
    parser.add_argument("--reset-completed", action="store_true",
                        help="启动时清空 completed_tasks 文件（等价于从零开始跑）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    completed_tasks_file = os.path.join(args.output_dir, f"completed_tasks_{args.folder}.txt")

    if args.reset_completed and os.path.exists(completed_tasks_file):
        os.remove(completed_tasks_file)

    completed_tasks = set()
    if os.path.exists(completed_tasks_file):
        with open(completed_tasks_file, "r") as f:
            completed_tasks = set(line.strip() for line in f if line.strip())

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    libero_root = os.path.dirname(current_script_dir)
    local_pkg_root = os.path.join(libero_root, "libero", "libero")

    folder_path = os.path.join(args.data_root, args.folder)
    if not os.path.exists(folder_path):
        print(f"[错误] 文件夹不存在: {folder_path}")
        return

    task_folders = sorted([d for d in os.listdir(folder_path)
                           if os.path.isdir(os.path.join(folder_path, d))])

    print(f"[info] 在 {args.folder} 中找到 {len(task_folders)} 个任务")
    print(f"[info] 已完成 {len(completed_tasks)} 个任务")

    # Ensure folder exists
    out_folder = os.path.join(args.output_dir, args.folder)
    os.makedirs(out_folder, exist_ok=True)

    # Track successes and rewrite completed file (avoid duplicates)
    succeeded = set(completed_tasks)

    for idx, task_name in enumerate(task_folders):
        demo_file = os.path.join(folder_path, task_name, "demo.hdf5")
        if not os.path.exists(demo_file):
            print(f"\n[{idx + 1}/{len(task_folders)}] [警告] demo文件不存在: {demo_file}，跳过")
            continue

        video_filename = f"{task_name}_traj{args.trajectory_index}.mp4"
        video_path = os.path.join(out_folder, video_filename)

        marked_done = (task_name in completed_tasks)
        has_video = _video_seems_ok(video_path)
        if (not args.force) and marked_done and has_video:
            print(f"\n[{idx + 1}/{len(task_folders)}] 任务 {task_name} 已完成且视频存在，跳过")
            continue
        if (not args.force) and marked_done and (not has_video):
            print(f"\n[{idx + 1}/{len(task_folders)}] 记录显示已完成，但视频不存在/过小：将重新处理 {task_name}")
        else:
            print(f"\n[{idx + 1}/{len(task_folders)}] 处理任务: {task_name}")

        try:
            ok = process_demo(
                demo_file=demo_file,
                trajectory_index=args.trajectory_index,
                video_path=video_path,
                use_camera_obs=args.use_camera_obs,
                use_depth=args.use_depth,
                local_pkg_root=local_pkg_root,
                cap_index=args.cap_index,
                action_pause=args.action_pause,
                video_fps=args.video_fps,
                no_display=args.no_display,
                mirror_lr=args.mirror_lr,
            )
            if ok:
                succeeded.add(task_name)
                print(f"[成功] 视频已保存到: {video_path}")
            else:
                print(f"[失败] 未生成有效视频: {task_name}")

        except Exception as e:
            print(f"[错误] 处理任务 {task_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    # Rewrite completed file
    with open(completed_tasks_file, "w") as f:
        for name in sorted(succeeded):
            f.write(name + "\n")

    print("\n[完成] 所有任务处理完毕！")
    print(f"[完成] completed_tasks 记录数: {len(succeeded)}")


if __name__ == "__main__":
    main()