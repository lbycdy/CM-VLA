import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundSawyer(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        # super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        print("Initializing Sawyer's joint positions:", np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57]))

        # return np.array(
        #     # [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        #     [0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57]       #
        #     # [np.pi/2, -np.pi/2, 0.00, np.pi/2, 0.00,np.pi/2, 0]
        #     # [0, -np.pi/2, 0.00, np.pi/2, 0.00,np.pi/2, 0]
        #
        # )

        # return np.array([-1.14567029e-03, -7.18644688e-02, -1.33746836e+00, 6.96581128e-01, 1.38168652e+00, 1.75589178e+00,
        #              3.95774444e+00])
        # return np.array(
        #     [-0.05000000, -1.20700000, -0.20200000, 1.91700000, 2.97600000, -0.84500000, 1.39500000])

        return np.array(
            [-0.05000000, -1.20700000, -0.20200000, 1.91700000, 2.27600000, -0.84500000, 1.39500000])
        # return np.array(
        #     [-0.1997, -0.1176, 0.0869, -0.4859, -0.1241, 1.7749, 1.5840])
        
    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0.42,
            ),
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
