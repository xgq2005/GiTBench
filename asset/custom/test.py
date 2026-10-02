import pybullet as p
import pybullet_data
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("./custom_structure.urdf", [0, 0, 0])
while True:
    p.stepSimulation()