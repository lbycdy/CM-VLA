import os
os.environ["MUJOCO_GL"] = "glfw"
import robosuite as suite

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,   # ✅ 只显示窗口，不取图像
)

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

