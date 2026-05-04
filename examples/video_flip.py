import argparse
import subprocess
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".flv",
    ".wmv", ".webm", ".m4v", ".ts", ".mts", ".m2ts"
}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def mirror_video(src: Path, dst: Path, overwrite: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        print(f"跳过，文件已存在：{dst}")
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i", str(src),

        # 保留所有输入流：视频、音频、字幕、附件等
        "-map", "0",

        # 只对视频做
        "-vf", "hflip", #左右
        # "-vf", "vflip",#上下

        # 视频重新编码
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",

        # 其他流直接复制
        "-c:a", "copy",
        "-c:s", "copy",
        "-c:d", "copy",

        # 保留元数据和章节
        "-map_metadata", "0",
        "-map_chapters", "0",

        str(dst)
    ]

    print(f"处理中：{src}")
    subprocess.run(cmd, check=True)
    print(f"完成：{dst}")


def main():
    parser = argparse.ArgumentParser(description="批量左右镜像指定目录下的视频文件")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/home/lbycdy/work/camerapi/examples/libero/datasets2",
        help="输入视频目录"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="/home/lbycdy/work/camerapi/examples/libero/datasets3",
        help="输出目录，默认是在输入目录同级创建 mirrored_videos"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，则覆盖"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="只处理当前目录，不递归子目录"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入路径不是有效目录：{input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.parent / "mirrored_videos"

    video_files = (
        input_dir.glob("*")
        if args.no_recursive
        else input_dir.rglob("*")
    )

    count = 0

    for src in video_files:
        if not is_video_file(src):
            continue

        relative_path = src.relative_to(input_dir)
        dst = output_dir / relative_path

        mirror_video(src, dst, overwrite=args.overwrite)
        count += 1

    print(f"\n全部完成，共处理 {count} 个视频。")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()