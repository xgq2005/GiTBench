import numpy as np
INDUSTRIAL_TASK_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            -0.615,
            0,
            0.55,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.065, 0, 0],
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "base_camera_width": 1280,
        "base_camera_height": 720,
        "sensor_fovx": np.deg2rad(90),
        "sensor_fovy": np.deg2rad(65),
        "left_arm_offset": [-0.615, 0.3, 0],
        "right_arm_offset": [-0.615, -0.3, 0],
        "single_arm_offset": [-0.615, 0, 0],
    },

}
