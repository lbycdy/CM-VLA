from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

from .mounted_sawyer import MountedSawyer
from .on_the_ground_sawyer import OnTheGroundSawyer

from .mounted_ur5 import MountedUR5e
from .on_the_ground_ur5e import OnTheGroundUR5e

from .mounted_jaco import MountedJaco
from .on_the_ground_jaco import OnTheGroundJaco

from .mounted_xarm import Mountedxarm
from .on_the_ground_xarm import OnTheGroundxarm

from .mounted_jaco import MountedJaco
from .on_the_ground_jaco import OnTheGroundJaco

from .mounted_iiwa import Mountediiwa
from .on_the_ground_iiwa import OnTheGroundiiwa

from .mounted_kinova3 import Mountedkinova3
from .on_the_ground_kinova3 import OnTheGroundkinova3

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "MountedUR5e": SingleArm,
        "OnTheGroundUR5e": SingleArm,
        "MountedJaco" : SingleArm,
        "OnTheGroundJaco" : SingleArm,
        "Mountedxarm" : SingleArm,
        "OnTheGroundxarm" : SingleArm,
        "Mountediiwa" : SingleArm,
        "OnTheGroundiiwa" : SingleArm,
        "Mountedkinova3" : SingleArm,
        "OnTheGroundkinova3" : SingleArm,
    }
)
# ROBOT_CLASS_MAPPING.update(
#     {
#         "MountedPanda": SingleArm,
#         "OnTheGroundPanda": SingleArm,
#     }
# )