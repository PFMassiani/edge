import unittest
import mujoco_py
import os

class TestMujoco(unittest.TestCase):
    def test_model_creation(self):
        mj_path, _ = mujoco_py.utils.discover_mujoco()
        xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
        model = mujoco_py.load_model_from_path(xml_path)
        sim = mujoco_py.MjSim(model)
        self.assertTrue(True)