import argparse
import os
import traceback

import cv2
import h5py
import numpy as np


def write_video_from_hdf5(
    hdf5_path,
    video_path,
    demo_name="demo_0",
    image_key="agentview_rgb",
    fps=20,
    skip_frames=0,
):
    """
    直接从 HDF5 的 data/<demo_name>/obs/<image_key> 读取图像并写成视频。
    不重建环境，不依赖 BDDL / robosuite / LIBERO 场景创建。
    """
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError("文件中缺少 'data' 组")

        data_group = f["data"]

        if demo_name not in data_group:
            raise KeyError(f"文件中缺少轨迹: {demo_name}，可用轨迹: {list(data_group.keys())}")

        demo_group = data_group[demo_name]

        if "obs" not in demo_group:
            raise KeyError(f"{demo_name} 中缺少 'obs' 组")

        obs_group = demo_group["obs"]

        if image_key not in obs_group:
            raise KeyError(
                f"obs 中缺少图像键 '{image_key}'，可用键: {list(obs_group.keys())}"
            )

        frames = obs_group[image_key]

        if len(frames.shape) != 4:
            raise ValueError(f"{image_key} 不是 4 维图像序列，shape={frames.shape}")

        if frames.shape[-1] != 3:
            raise ValueError(f"{image_key} 不是 RGB 图像，shape={frames.shape}")

        num_frames, height, width, channels = frames.shape
        if num_frames <= skip_frames:
            raise ValueError(
                f"总帧数 {num_frames} 不足以跳过 {skip_frames} 帧"
            )

        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

        written = 0
        try:
            for i in range(skip_frames, num_frames):
                frame = np.asarray(frames[i])[::-1, ::-1]

                # HDF5 里存的是 RGB，OpenCV 写视频要 BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                written += 1
        finally:
            writer.release()

        if written == 0:
            raise RuntimeError("没有写入任何视频帧")

        return {
            "num_frames_total": int(num_frames),
            "num_frames_written": int(written),
            "height": int(height),
            "width": int(width),
        }


def main():
    parser = argparse.ArgumentParser(description="直接从 HDF5 中提取 RGB 图像并生成视频")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/datasets/sawyer",
        help="数据根目录",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_object","libero_spatial"],
        help="数据子目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/lbycdy/work/camerapi/video_show",
        help="视频输出目录",
    )
    parser.add_argument(
        "--demo-name",
        type=str,
        default="demo_0",
        help="要导出的视频轨迹名，例如 demo_0",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="agentview_rgb",
        help="obs 里的图像键，例如 agentview_rgb 或 eye_in_hand_rgb",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="输出视频帧率",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="跳过前几帧",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只处理第一个 hdf5 文件，便于测试",
    )

    args = parser.parse_args()

    folder_path = os.path.join(args.data_root, args.folder)
    if not os.path.exists(folder_path):
        print(f"[错误] 文件夹不存在: {folder_path}")
        return

    demo_files = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith(".hdf5")
            and os.path.isfile(os.path.join(folder_path, f))
        ]
    )

    print(f"[info] 数据目录: {folder_path}")
    print(f"[info] 找到 {len(demo_files)} 个 hdf5 文件")
    print(f"[info] 使用 demo: {args.demo_name}")
    print(f"[info] 使用图像键: {args.image_key}")

    if len(demo_files) == 0:
        print("[警告] 没有找到任何 .hdf5 文件")
        return

    success_count = 0
    fail_count = 0

    iterable = demo_files[:1] if args.only_one else demo_files

    for idx, filename in enumerate(iterable, start=1):
        hdf5_path = os.path.join(folder_path, filename)
        stem = os.path.splitext(filename)[0]
        video_name = f"{stem}_{args.demo_name}_{args.image_key}.mp4"
        video_path = os.path.join(args.output_dir, args.folder, video_name)

        print(f"\n[{idx}/{len(iterable)}] 处理文件: {filename}")

        try:
            info = write_video_from_hdf5(
                hdf5_path=hdf5_path,
                video_path=video_path,
                demo_name=args.demo_name,
                image_key=args.image_key,
                fps=args.fps,
                skip_frames=args.skip_frames,
            )
            success_count += 1
            print(f"[成功] 视频已保存到: {video_path}")
            print(
                f"[成功] 写入 {info['num_frames_written']} 帧 "
                f"(总帧数 {info['num_frames_total']}，分辨率 {info['width']}x{info['height']})"
            )
        except Exception as e:
            fail_count += 1
            print(f"[错误] 处理失败: {filename}")
            print(f"[错误] 原因: {e}")
            traceback.print_exc()

    print("\n[完成] 全部处理结束")
    print(f"[完成] 成功: {success_count}")
    print(f"[完成] 失败: {fail_count}")


if __name__ == "__main__":
    main()