import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Tuple, Optional

import imageio
import numpy as np
import tqdm
import tyro
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path

# Optional: some repos require this to set up sys.path (same as collect_demonstration.py)
try:
    import init_path  # noqa: F401
except Exception:
    pass

# Fallback env (old behavior)
from libero.libero.envs import OnScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO evaluation parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 5
    num_trials_per_task: int = 5
    video_out_path: str = "data/libero"
    seed: int = 7

    #################################################################################################################
    # Environment init (match collect_demonstration.py style)
    #################################################################################################################
    # Choose backend:
    # - "task_mapping": create env via TASK_MAPPING[problem_name](...), like collect_demonstration.py
    # - "onscreen": old behavior (OnScreenRenderEnv)
    env_backend: str = "task_mapping"

    # robosuite / libero env kwargs (mirrors collect_demonstration.py, but we enable camera obs for policy eval)
    robots: Tuple[str, ...] = ("UR5e",)  # change to ("Panda",) if your libero tasks are Panda-based
    controller: str = "OSC_POSE"
    env_configuration: str = "single-arm-opposed"  # only used for TwoArm tasks
    control_freq: int = 20
    reward_shaping: bool = True
    ignore_done: bool = True

    # rendering & camera obs
    has_renderer: bool = True                  # on-screen viewer
    render_camera: str = "agentview"           # on-screen camera
    use_camera_obs: bool = True                # must be True if your model consumes images
    camera_names: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    save_first_frame: bool = True
    first_frame_cam_name: str = "first_frame_agentview.png"
    first_frame_wrist_name: str = "first_frame_wrist.png"

def _quat2axisangle(quat):
    """Copied from robosuite transform utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution: int, seed: int, args: Args):
    """
    Create env in a way that matches collect_demonstration.py:
      problem_info = BDDLUtils.get_problem_info(bddl_file)
      env = TASK_MAPPING[problem_name](bddl_file_name=..., robots=..., controller_configs=..., ...)
    """
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    # task description: prefer BDDL language_instruction when available; fallback to task.language
    task_description = getattr(task, "language", str(task_bddl_file))

    if args.env_backend == "onscreen":
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
        env = OnScreenRenderEnv(**env_args)
        if hasattr(env, "seed"):
            env.seed(seed)
        return env, task_description

    # task_mapping backend
    try:
        import libero.libero.envs.bddl_utils as BDDLUtils
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config
    except Exception as e:
        logging.warning(f"Failed to import TASK_MAPPING backend deps ({e}); falling back to OnScreenRenderEnv.")
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
        env = OnScreenRenderEnv(**env_args)
        if hasattr(env, "seed"):
            env.seed(seed)
        return env, task_description

    assert task_bddl_file.exists(), f"BDDL file not found: {task_bddl_file}"
    problem_info = BDDLUtils.get_problem_info(str(task_bddl_file))
    problem_name = problem_info["problem_name"]
    task_description = problem_info.get("language_instruction", task_description)

    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": list(args.robots),
        "controller_configs": controller_config,
    }
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.env_configuration

    env_kwargs = dict(
        bddl_file_name=str(task_bddl_file),
        **config,
        has_renderer=args.has_renderer,
        has_offscreen_renderer=bool(args.use_camera_obs),
        render_camera=args.render_camera,
        ignore_done=args.ignore_done,
        use_camera_obs=args.use_camera_obs,
        reward_shaping=args.reward_shaping,
        control_freq=args.control_freq,
    )

    if args.use_camera_obs:
        cam_names = list(args.camera_names)
        env_kwargs.update(
            camera_names=cam_names,
            camera_heights=[resolution] * len(cam_names),
            camera_widths=[resolution] * len(cam_names),
        )

    env = TASK_MAPPING[problem_name](**env_kwargs)

    # seed if supported
    if hasattr(env, "seed"):
        env.seed(seed)

    return env, task_description


def _is_success(env, done, info) -> bool:
    """Robust success check across different env backends."""
    if hasattr(env, "_check_success"):
        try:
            return bool(env._check_success())
        except Exception:
            pass
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    return bool(done)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 420
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    first_frame_saved = False
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            try:
                obs = env.reset()
            except TypeError:
                # some wrappers don't return obs on reset
                env.reset()
                obs = None

            action_plan = collections.deque()

            # Set initial states if the backend supports it (OnScreenRenderEnv does; TASK_MAPPING backend may or may not)
            if hasattr(env, "set_init_state"):
                try:
                    obs = env.set_init_state(initial_states[episode_idx])
                except Exception as e:
                    logging.warning(f"set_init_state failed ({e}); continuing with reset() state.")

            t = 0
            replay_images = []
            success = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        if args.has_renderer:
                            try:
                                env.render()
                            except Exception:
                                pass
                        t += 1
                        continue

                    # Image preprocessing (expects obs contains camera images)
                    if obs is None:
                        raise RuntimeError("obs is None; your env.reset() / set_init_state() didn't return observations.")

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                    if t == args.num_steps_wait:
                        out_dir = Path.cwd()  # 项目根路径：确保你在项目根目录运行
                        cam_path = out_dir / "first_frame_agentview.png"
                        wrist_path = out_dir / "first_frame_wrist.png"

                        imageio.imwrite(cam_path, img)
                        imageio.imwrite(wrist_path, wrist_img)

                        print(f"Saved agentview to: {cam_path.resolve()}")
                        print(f"Saved wrist to: {wrist_path.resolve()}")

                    replay_images.append(img)

                    if not action_plan:
                        gq = np.asarray(obs["robot0_gripper_qpos"]).ravel()
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), gq[[1, 3]])
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan_steps={args.replan_steps}, but policy predicts only {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if args.has_renderer:
                        try:
                            env.render()
                        except Exception:
                            pass

                    success = _is_success(env, done, info)
                    if success:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video
            suffix = "success" if success else "failure"
            task_segment = str(task_description).replace(" ", "_")
            if replay_images:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # close per-task env to avoid viewer / mujoco resource leaks
        try:
            env.close()
        except Exception:
            pass

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)