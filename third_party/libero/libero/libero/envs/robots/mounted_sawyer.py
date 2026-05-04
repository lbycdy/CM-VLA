import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


# class MountedPanda(ManipulatorModel):
#     """
#     Panda is a sensitive single-arm robot designed by Franka.
#     Args:
#         idn (int or str): Number or some other unique identification string for this robot instance
#     """
#
#     def __init__(self, idn=0):
#         # super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)
#         super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)
#
#         # Set joint damping
#         self.set_joint_attribute(
#             attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
#         )
#
#     @property
#     def default_mount(self):
#         return "RethinkMount"
#
#
#
#     @property
#     def default_gripper(self):
#         return "RethinkGripper"
#
#     @property
#     def default_controller_config(self):
#         return "default_sawyer"
#
#     @property
#     def init_qpos(self):
#         return np.array(
#             [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
#         )
#
#     @property
#     def base_xpos_offset(self):
#         return {
#             "bins": (-0.5, -0.1, 0),
#             "empty": (-0.6, 0, 0),
#             "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#             "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
#             "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#         }
#
#     @property
#     def top_offset(self):
#         return np.array((0, 0, 1.0))
#
#     @property
#     def _horizontal_radius(self):
#         return 0.5
#
#     @property
#     def arm_type(self):
#         return "single"

class MountedSawyer(ManipulatorModel):
    """
    Sawyer is a witty single-arm robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """



    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)
        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        # print('-------------------------------------')
        # exit()
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])
        # return np.array( [-2.62945473e-05, -1.32536805e+00 ,-2.37050993e-01 , 2.16586483e+00,
  # 5.18503486e-02 , 7.16823974e-01 ,-1.84341972e+00]     )

     #    return np.array([-7.06636465e-04, -1.04639152e+00 , 2.45355371e-01 , 1.41767979e+00,
     # 2.56752267e+00, -1.05003230e+00,  2.03854604e+00])
     #    return np.array([-1.14567029e-03, -7.18644688e-02, -1.33746836e+00, 6.96581128e-01, 1.38168652e+00, 1.75589178e+00, 3.95774444e+00])

        return np.array(
            [-0.22988284, -0.90734066, 0.15766940, 1.13659740, -0.27610000, 1.37533650, 1.75484790])



    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"