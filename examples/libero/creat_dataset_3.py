import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import cv2

import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

# import init_path
import libero.libero.utils.utils as libero_utils

from libero.libero.envs import *
from glob import glob as _glob


import re

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


def expand_demo_inputs(inputs):
    """
    Expand --demo-file inputs into a list of demo.hdf5 files.

    Supports:
      - multiple explicit files
      - directories (recursive search for demo.hdf5 under them)
      - glob patterns (e.g. /path/*/demo.hdf5)
    """
    results = []
    seen = set()

    for x in inputs:
        x = os.path.expanduser(x)

        # expand glob if needed
        if any(ch in x for ch in ["*", "?", "["]):
            candidates = _glob(x, recursive=True)
        else:
            candidates = [x]

        for c in candidates:
            p = Path(c)
            if p.is_dir():
                for f in sorted(p.rglob("demo.hdf5")):
                    s = str(f)
                    if s not in seen:
                        results.append(s)
                        seen.add(s)
            elif p.is_file():
                s = str(p)
                if s not in seen:
                    results.append(s)
                    seen.add(s)
            else:
                print(f"[warn] not found: {c}")

    return results


def _get_rel_out_from_bddl(bddl_file_name: str) -> str:
    """
    Make output path relative to bddl_files/...
    Example:
      /xxx/bddl_files/libero_goal/open_xxx.bddl  -> libero_goal/open_xxx_demo.hdf5
    If bddl_files/ not found, fallback to basename folder.
    """
    bddl_file_dir = os.path.dirname(bddl_file_name)
    if "bddl_files/" in bddl_file_dir:
        rel_dir = bddl_file_dir.split("bddl_files/")[-1]
    else:
        rel_dir = os.path.basename(bddl_file_dir)

    out_name = os.path.basename(bddl_file_name).replace(".bddl", "_demo.hdf5")
    return os.path.join(rel_dir, out_name)


def _next_demo_index(grp) -> int:
    """
    Return next available demo index in grp "data".
    grp has keys like demo_0, demo_1, ...
    """
    max_i = -1
    for k in grp.keys():
        if not k.startswith("demo_"):
            continue
        try:
            i = int(k.split("_", 1)[1])
            max_i = max(max_i, i)
        except Exception:
            pass
    return max_i + 1


def _set_attr_if_missing(grp, key, value):
    if key not in grp.attrs:
        grp.attrs[key] = value


def get_camera_extrinsic(cam_xmat, cam_xpos):
    """
    Construct camera extrinsic matrix from rotation matrix and position.
    
    Args:
        cam_xmat: 3x3 rotation matrix
        cam_xpos: 3x1 position vector
    
    Returns:
        4x4 extrinsic matrix [R | t]
                            [0 0 0 1]
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = cam_xmat.reshape(3, 3)
    extrinsic[:3, 3] = cam_xpos
    return extrinsic


def process_one_demo(args, demo_file: str):
    """
    Read one input demo.hdf5 (raw robosuite collection),
    and APPEND all its episodes into the output file for the SAME TASK (same BDDL).
    """
    demo_file = os.path.expanduser(demo_file)
    print(f"\n=== Processing input demo: {demo_file} ===")

    f_in = h5py.File(demo_file, "r")
    if "data" not in f_in:
        raise KeyError(f"Invalid demo file (no /data group): {demo_file}")

    env_name = f_in["data"].attrs.get("env", "")
    env_kwargs = json.loads(f_in["data"].attrs["env_info"])
    problem_info = json.loads(f_in["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    bddl_file_name = f_in["data"].attrs["bddl_file_name"]
    
    # Fix bddl_file_name path if it points to a different user's directory
    if "bddl_files/" in bddl_file_name:
        # Extract the relative path after bddl_files/
        bddl_relative = bddl_file_name.split("bddl_files/", 1)[1]
        # Get the current script's libero path
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        libero_root = os.path.dirname(current_script_dir)  # Go up from scripts/ to libero/
        # Reconstruct the correct path
        bddl_file_name = os.path.join(libero_root, "libero", "libero", "bddl_files", bddl_relative)
        print(f"[info] Corrected bddl path to: {bddl_file_name}")

    # episodes in the input demo file
    demos = [k for k in f_in["data"].keys() if k.startswith("demo_")]
    demos = sorted(demos, key=lambda x: int(x.split("_", 1)[1]) if x.split("_", 1)[1].isdigit() else x)

    # -----------------------------
    # OUTPUT PATH: same task -> same output file
    # -----------------------------
    dataset_root = os.path.expanduser(args.dataset_path)
    rel_out = _get_rel_out_from_bddl(bddl_file_name)
    out_path = os.path.join(dataset_root, rel_out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # open output in append mode (do NOT overwrite)
    out_mode = "a" if os.path.exists(out_path) else "w"
    f_out = h5py.File(out_path, out_mode)

    if "data" in f_out:
        grp = f_out["data"]
    else:
        grp = f_out.create_group("data")

    # initialize / keep metadata (only set if missing)
    _set_attr_if_missing(grp, "env_name", env_name)
    _set_attr_if_missing(grp, "problem_info", f_in["data"].attrs["problem_info"])
    _set_attr_if_missing(grp, "macros_image_convention", macros.IMAGE_CONVENTION)

    _set_attr_if_missing(grp, "bddl_file_name", bddl_file_name)
    try:
        _set_attr_if_missing(grp, "bddl_file_content", open(bddl_file_name, "r").read())
    except Exception as e:
        print(f"[warn] cannot read bddl file content: {bddl_file_name} ({e})")

    # update env kwargs for playback
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=["robot0_eye_in_hand", "agentview"],
        reward_shaping=True,
        control_freq=20,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations=None,
    )

    # keep env_args if missing
    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_file_name,
        "env_kwargs": env_kwargs,
    }
    _set_attr_if_missing(grp, "env_args", json.dumps(env_args))

    # Create environment (for playback)
    env = TASK_MAPPING[problem_name](**env_kwargs)

    cap_index = 5
    total_added = 0
    demo_idx = _next_demo_index(grp)

    for ep in demos:
        print(f"Playing back episode {ep} -> will write as demo_{demo_idx} (press ESC to quit windows)")

        try:
            model_xml = f_in[f"data/{ep}"].attrs["model_file"]
        except Exception as e:
            print(f"[error] Cannot read model_file from {ep}: {e}")
            continue

        # robust reset with timeout
        reset_success = False
        reset_attempts = 0
        max_reset_attempts = 10
        while not reset_success and reset_attempts < max_reset_attempts:
            try:
                env.reset()
                reset_success = True
            except Exception as e:
                reset_attempts += 1
                print(f"[warning] Reset attempt {reset_attempts} failed: {e}")
                if reset_attempts >= max_reset_attempts:
                    print(f"[error] Failed to reset environment after {max_reset_attempts} attempts, skipping {ep}")
                    break
        
        if not reset_success:
            continue

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        local_pkg_root = os.path.join(libero_root, "libero", "libero")
        model_xml = remap_mujoco_asset_paths(model_xml, local_pkg_root)

        if not args.use_camera_obs and getattr(env, "viewer", None) is not None:
            env.viewer.set_camera(0)

        try:
            states = f_in[f"data/{ep}/states"][()]
            actions = np.array(f_in[f"data/{ep}/actions"][()])
        except Exception as e:
            print(f"[error] Cannot read states/actions from {ep}: {e}")
            continue

        num_actions = actions.shape[0]
        init_idx = 0

        try:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()
        except Exception as e:
            print(f"[error] Failed to reset from XML or set initial state for {ep}: {e}")
            continue

        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []

        agentview_images = []
        eye_in_hand_images = []
        agentview_depths = []
        eye_in_hand_depths = []

        valid_index = []
        
        # Extract camera extrinsic matrices at the beginning (after env setup)
        agentview_extrinsic = np.eye(4)
        eye_in_hand_extrinsic = np.eye(4)
        
        if args.use_camera_obs:
            try:
                # agentview camera (index 2)
                agentview_xmat = env.sim.data.cam_xmat[2].copy()
                agentview_xpos = env.sim.data.cam_xpos[2].copy()
                agentview_extrinsic = get_camera_extrinsic(agentview_xmat, agentview_xpos)
                
                # robot0_eye_in_hand camera (index 6)
                eye_in_hand_xmat = env.sim.data.cam_xmat[6].copy()
                eye_in_hand_xpos = env.sim.data.cam_xpos[6].copy()
                eye_in_hand_extrinsic = get_camera_extrinsic(eye_in_hand_xmat, eye_in_hand_xpos)
                
                print(f"[info] Extracted camera extrinsics for {ep}")
            except Exception as e:
                print(f"[warning] Failed to extract camera extrinsics for {ep}: {e}, using identity matrices")

        for j, action in enumerate(actions):
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                print(f"[error] Step {j} failed for {ep}: {e}")
                break

            if j < num_actions - 1:
                state_playback = env.sim.get_state().flatten()
                err = np.linalg.norm(states[j + 1] - state_playback)
                if err > 0.01:
                    print(f"[warning] playback diverged by {err:.2f} for {ep} at step {j}")

            if j < cap_index:
                continue

            valid_index.append(j)

            if not args.no_proprio:
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                )

            robot_states.append(env.get_robot_state_vector(obs))

            if args.use_camera_obs:
                if args.use_depth:
                    depth_agent = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
                    depth_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_depth"][::-1, ::-1])
                    agentview_depths.append(depth_agent)
                    eye_in_hand_depths.append(depth_wrist)
                    # agentview_depths.append(obs["agentview_depth"])
                    # eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                eye_in_hand_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                agentview_images.append(agentview_image)
                eye_in_hand_images.append(eye_in_hand_image)

                # # show (RGB->BGR)
                # img_a = cv2.cvtColor(agentview_image, cv2.COLOR_RGB2BGR)
                # img_e = cv2.cvtColor(eye_in_hand_image, cv2.COLOR_RGB2BGR)

                # cv2.imshow("agentview", img_a)
                # cv2.imshow("eye", img_e)

                # # allow ESC to close visualization early (still continues saving)
                # k = cv2.waitKey(1) & 0xFF
                # if k == 27:  # ESC
                #     cv2.destroyAllWindows()
            else:
                env.render()

        # Skip if no valid data collected
        if len(valid_index) == 0:
            print(f"[warning] No valid data collected for {ep}, skipping")
            continue

        # cut by valid_index
        states = states[valid_index]
        actions = actions[valid_index]

        dones = np.zeros(len(actions), dtype=np.uint8)
        rewards = np.zeros(len(actions), dtype=np.uint8)
        if len(actions) > 0:
            dones[-1] = 1
            rewards[-1] = 1

        # write one demo group
        ep_data_grp = grp.create_group(f"demo_{demo_idx}")
        demo_idx += 1

        obs_grp = ep_data_grp.create_group("obs")
        if not args.no_proprio and len(joint_states) > 0:
            if len(gripper_states) > 0:
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

        if len(agentview_images) > 0:
            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))

        if args.use_depth and len(agentview_depths) > 0:
            obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
            obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))

        # Save camera extrinsic matrices (parallel to obs group)
        ep_data_grp.create_dataset("agentview_extrinsic", data=agentview_extrinsic)
        ep_data_grp.create_dataset("eye_in_hand_extrinsic", data=eye_in_hand_extrinsic)

        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0) if len(robot_states) > 0 else np.zeros((0,)))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)

        ep_data_grp.attrs["num_samples"] = int(len(agentview_images))
        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["init_state"] = states[init_idx] if len(states) > 0 else np.zeros((0,))

        total_added += int(len(agentview_images))
        print(f"[success] Saved {ep} as demo_{demo_idx-1} with {len(agentview_images)} samples")

    # update global counters (based on actual groups)
    all_demos = [k for k in grp.keys() if k.startswith("demo_")]
    grp.attrs["num_demos"] = int(len(all_demos))
    grp.attrs["total"] = int(grp.attrs.get("total", 0)) + int(total_added)

    env.close()
    cv2.destroyAllWindows()

    f_out.close()
    f_in.close()

    print(f"[done] appended {len(demos)} episodes from {demo_file}")
    print("The created dataset is saved in the following path:")
    print(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-file",
        nargs="+",
        default=["/home/lbycdy/work/camerapi/examples/libero/demonstration_data/spatial/"],
        help="One or more demo.hdf5 paths, or directories containing demo.hdf5 (recursive), or glob patterns.",
    )
    parser.add_argument("--use-actions", action="store_true")
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
        help="Output root directory. Same task (same BDDL) will merge into the same output file under this root.",
    )
    parser.add_argument("--dataset-name", type=str, default="training_set")  # kept for compatibility (not used)
    parser.add_argument("--no-proprio", action="store_true")
    parser.add_argument("--use-depth", action="store_true")

    args = parser.parse_args()

    demo_files = expand_demo_inputs(args.demo_file)
    if not demo_files:
        raise FileNotFoundError("No demo.hdf5 files found from --demo-file inputs.")

    print("Found demo files:")
    for p in demo_files:
        print("  -", p)

    for demo_file in demo_files:
        process_one_demo(args, demo_file)


if __name__ == "__main__":
    main()
