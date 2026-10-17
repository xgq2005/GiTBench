"""
生物实验室场景配置文件
统一管理不同生物实验室场景的参数配置，方便切换不同场景时统一应用
"""
import numpy as np
from mani_skill.envs.tasks.gitbench_paths import project_asset_path

# 生物实验室场景配置
BIOLAB_SCENE_CONFIGS = {
    # ========== L2 场景：新样本容器任务 ==========
    "L2_01": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        "max_episode_steps": 200,
        
        # 相机配置
        "base_camera_width": 640,
        "base_camera_height": 480,
        "sensor_cam_eye_pos": [0.5, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "sensor_fovx": np.deg2rad(69.4),
        "sensor_fovy": np.deg2rad(42.5),
        
        # 双臂机器人生位姿
        "left_robot_pose": [-0.5, -0.2, 0],
        "right_robot_pose": [-0.5, 0.2, 0],
    },
    
    # ========== L4 场景：搅拌计数记忆任务 ==========
    "L4_01": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        "max_episode_steps": 200,
        
        # 相机配置
        "base_camera_width": 640,
        "base_camera_height": 480,
        "sensor_cam_eye_pos": [0.5, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "sensor_fovx": np.deg2rad(69.4),
        "sensor_fovy": np.deg2rad(42.5),
        
        # 双臂机器人生位姿
        "left_robot_pose": [-0.5, 0.3, 0],
        "right_robot_pose": [-0.5, -0.3, 0],
    },
    
    # ========== L5 场景：样本传递召回任务 ==========
    "L5_01": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        "max_episode_steps": 200,
        
        # 相机配置
        "base_camera_width": 640,
        "base_camera_height": 480,
        "sensor_cam_eye_pos": [0.5, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "sensor_fovx": np.deg2rad(69.4),
        "sensor_fovy": np.deg2rad(42.5),
        
        # 双臂机器人生位姿
        "left_robot_pose": [-0.5, -0.2, 0],
        "right_robot_pose": [-0.5, 0.2, 0],
    },
    
    # ========== L6 场景：反向盖子关闭任务 ==========
    "L6_01": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        "max_episode_steps": 200,
        
        # 相机配置
        "base_camera_width": 640,
        "base_camera_height": 480,
        "sensor_cam_eye_pos": [0.5, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "sensor_fovx": np.deg2rad(69.4),
        "sensor_fovy": np.deg2rad(42.5),
        
        # 双臂机器人生位姿
        "left_robot_pose": [-0.5, -0.2, 0],
        "right_robot_pose": [-0.5, 0.2, 0],
    },
    
    # ========== PiperX 单臂配置（参考） ==========
    "piper_x_single": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        
        # 相机配置
        "base_camera_width": 640,
        "base_camera_height": 480,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        "sensor_fovx": np.deg2rad(69.4),
        "sensor_fovy": np.deg2rad(42.5),
    },
    
    # ========== 默认通用配置 ==========
    "default": {
        # 基础任务参数
        "goal_thresh": 0.025,
        "robot_init_qpos_noise": 0.02,
        "max_episode_steps": 200,
        
        # 相机配置
        "base_camera_width": 1280,
        "base_camera_height": 720,
        "sensor_cam_eye_pos": [0.5, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [-0.58,0,0.55], 
        "human_cam_target_pos": [-0.03,0,0],
        "sensor_fovx": np.deg2rad(87),
        "sensor_fovy": np.deg2rad(58),
        
        # 默认双臂位姿
        "left_robot_pose": [-0.55, 0.3, 0],
        "right_robot_pose": [-0.55, -0.3, 0],
    }
}

# 路径配置（可以根据实际环境修改）
BIOLAB_PATH_CONFIGS = {
    "chemistry_model_dir": str(project_asset_path("chemistry", "models")),
    "industry_model_dir": str(project_asset_path("industry", "models")),
}

def get_biolab_config(scene_name: str):
    """
    获取生物实验室场景配置，如果场景不存在则返回默认配置
    
    Args:
        scene_name: 场景名称，如 "L4_01", "L5_01" 等
    
    Returns:
        场景配置字典
    """
    if scene_name in BIOLAB_SCENE_CONFIGS:
        return BIOLAB_SCENE_CONFIGS[scene_name]
    else:
        print(f"[Warning] Scene {scene_name} not found in BIOLAB_SCENE_CONFIGS, using default config")
        return BIOLAB_SCENE_CONFIGS["default"]
