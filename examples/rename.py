from pathlib import Path
import re

folder = Path("/home/lbycdy/work/camerapi/examples/libero/datasets3")  # 当前目录

#pattern = re.compile(r"^rollout_(.+)_(failure|success)(\.[^.]+)$")
pattern = re.compile(r"_(demo_2_render)(\.[^.]+)$")

for file in folder.iterdir():
    if not file.is_file():
        continue

    match = pattern.match(file.name)
    if not match:
        continue

    new_name = f"{match.group(1)}_sawyer{match.group(3)}"
    new_path = file.with_name(new_name)

    print(f"{file.name}  ->  {new_name}")

    if new_path.exists():
        print(f"跳过：目标文件已存在 {new_name}")
        continue

    file.rename(new_path)