# /home/ningkang/new_drive/maniskill3/mani_skill/envs/tasks/custom_task/biolab/data_recorder.py

import os
import time
import h5py
import cv2
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import quat2euler


class BioLabDataRecorder:
    """
    生物实验室数据采集器 - 输出与实体机械臂相同的HDF5格式
    
    数据结构:
    arm/
    ├── endPose/
    │   ├── Left  (shape: [N, 6], dtype: float64)  # x,y,z,roll,pitch,yaw
    │   └── Right (shape: [N, 6], dtype: float64)
    ├── jointStatePosition/
    │   ├── Left  (shape: [N, J], dtype: float64)
    │   └── Right (shape: [N, J], dtype: float64)
    └── jointStateVelocity/
        ├── Left  (shape: [N, J], dtype: float64)
        └── Right (shape: [N, J], dtype: float64)
    camera/
    ├── color/
    │   └── <camera_name> (shape: [N], dtype: vlen uint8)
    └── depth/
        └── <camera_name> (shape: [N], dtype: vlen uint16)
    instruction        (空字符串)
    timestamp          (shape: [N], dtype: float64)
    size               (shape: [], dtype: int64)
    """
    
    # 相机名称映射表
    CAMERA_NAME_MAPPING = {
        'human_cam': 'front',
        'piper_x-0-hand_camera': 'left',
        'piper_x-1-hand_camera': 'right'
    }
    
    def __init__(self):
        # 机械臂名称（带puppet前缀，与episode_to_hdf5.py格式一致）
        # agent.agents[0] 对应 piper_x-0 (right), agent.agents[1] 对应 piper_x-1 (left)
        self.ARM_NAMES = ("puppetLeft","puppetRight")
        # 末端位姿字段顺序
        self.END_POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
        
        # 重置数据存储
        self.reset()
    
    def reset(self):
        """重置采集数据"""
        self.end_pose = {arm: [] for arm in self.ARM_NAMES}
        self.position = {arm: [] for arm in self.ARM_NAMES}
        self.velocity = {arm: [] for arm in self.ARM_NAMES}
        self.effort = {arm: [] for arm in self.ARM_NAMES}  # 关节effort（与episode_to_hdf5.py格式一致）
        self.color_bytes = {}  # 动态添加相机
        self.depth_bytes = {}  # 动态添加相机
        self.color_timestamps = {}  # 记录每张图片的时间戳
        self.depth_timestamps = {}  # 记录每张深度图的时间戳
        self.timestamps = []
        self.subtasks = []
        self.subtask_boundaries = []
        self.instruction = ""
    
    def set_subtask(self, subtask_label):
        """
        设置当前子任务标签，并记录帧边界
        
        Args:
            subtask_label: 子任务描述文本
        """
        self.subtasks.append(subtask_label)
        self.subtask_boundaries.append(len(self.timestamps))

    def export_subtask_csv(self, csv_path):
        """
        导出子任务帧范围到CSV文件
        合并连续的相同 subtask，避免重复输出
        
        Args:
            csv_path: CSV文件保存路径
        """
        import csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        total_frames = len(self.timestamps)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_range', 'subtask'])
            
            prev_label = None
            merged_start = None
            
            for i, (label, start_frame) in enumerate(zip(self.subtasks, self.subtask_boundaries)):
                if label == prev_label:
                    # 与上一个标签相同，继续合并
                    pass
                else:
                    # 标签变化，输出上一个合并段
                    if prev_label is not None:
                        writer.writerow([f'frame{merged_start}-frame{start_frame}', prev_label])
                    # 开始新的合并段
                    prev_label = label
                    merged_start = start_frame
            
            # 输出最后一个合并段
            if prev_label is not None:
                writer.writerow([f'frame{merged_start}-frame{total_frames}', prev_label])
        
        print(f"子任务CSV已保存到: {csv_path}")

    def capture_robot_state(self, agent):
        """
        采集机械臂状态数据
        
        Args:
            agent: 多智能体对象，包含双臂
        """
        timestamp = time.time()
        self.timestamps.append(timestamp)
        
        for idx, robot in enumerate(agent.agents):
            if idx >= len(self.ARM_NAMES):
                break  # 只处理双臂
            
            arm_name = self.ARM_NAMES[idx]
            
            # 关节状态（与episode_to_hdf5.py格式一致）
            try:
                robot_qpos = robot.robot.get_qpos()
                robot_qvel = robot.robot.get_qvel()
                
                # 处理可能的batch维度
                if len(robot_qpos.shape) == 2:
                    robot_qpos = robot_qpos[0]  # 取第一个环境
                if len(robot_qvel.shape) == 2:
                    robot_qvel = robot_qvel[0]  # 取第一个环境
                
                # 转换为numpy数组
                qpos_np = np.array(robot_qpos.cpu().numpy() if hasattr(robot_qpos, 'cpu') else robot_qpos, dtype=np.float64)
                qvel_np = np.array(robot_qvel.cpu().numpy() if hasattr(robot_qvel, 'cpu') else robot_qvel, dtype=np.float64)
                
                # 关节角度只保留7个（6个关节 + 1个夹具角度）
                if len(qpos_np) >= 7:
                    qpos_np = qpos_np[:7]
                    # 第7个数据：夹具角度改为两个夹具张开的完整距离（原为单个夹爪半边距离，乘以2）
                    qpos_np[6] = qpos_np[6] * 2
                
                # 关节速度只保留6个（去掉夹具速度）
                if len(qvel_np) >= 6:
                    qvel_np = qvel_np[:6]
                
                # 确保是1D数组
                if qpos_np.ndim == 2 and qpos_np.shape[0] == 1:
                    qpos_np = qpos_np[0]
                if qvel_np.ndim == 2 and qvel_np.shape[0] == 1:
                    qvel_np = qvel_np[0]
                
                self.position[arm_name].append(qpos_np)
                self.velocity[arm_name].append(qvel_np)
                
                # 关节effort（暂不采集，保持空列表）
                if arm_name not in self.effort:
                    self.effort[arm_name] = []
                self.effort[arm_name].append(np.array([], dtype=np.float64))
                
                # 末端位姿（转换为 xyz + rpy 格式，在机械臂基座坐标系下）
                # 使用 Link6（法兰盘），而不是 piper_x_hand_tcp（TCP 有额外 rpy="0 0 -1.5708" 旋转）
                # 师兄定义：末端 = 法兰盘，Z轴垂直法兰中心的右手系
                from mani_skill.utils import sapien_utils
                flange_link = sapien_utils.get_obj_by_name(robot.robot.get_links(), "Link6")
                ee_pose_world = flange_link.pose  # 世界坐标系下的法兰盘位姿
                
                # 获取机械臂基座位姿
                base_pose_world = robot.robot.pose  # 基座在世界系下的位姿
                
                # 将末端位姿从世界坐标系转换到基座坐标系
                # base_T_ee = base_T_world * world_T_ee = base_pose_world.inv() * ee_pose_world
                ee_pose_base = base_pose_world.inv() * ee_pose_world
                
                ee_p = ee_pose_base.p
                ee_q = ee_pose_base.q
                
                if len(ee_p.shape) == 2:
                    ee_p = ee_p[0]
                if len(ee_q.shape) == 2:
                    ee_q = ee_q[0]
                
                ee_p_np = np.array(ee_p.cpu().numpy() if hasattr(ee_p, 'cpu') else ee_p, dtype=np.float64)
                ee_q_np = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float64)
                
                # 将四元数转换为欧拉角（roll, pitch, yaw）
                # 使用 transforms3d 的 sxyz (static XYZ, 外旋) 约定，对齐真实机器人数据格式
                # SAPIEN 四元数格式为 [w, x, y, z]，transforms3d 也期望 [w, x, y, z]
                roll, pitch, yaw = quat2euler(ee_q_np, axes='sxyz')
                
                # 组合成6维位姿向量（基座坐标系下）
                pose_vec = np.array([ee_p_np[0], ee_p_np[1], ee_p_np[2], roll, pitch, yaw], dtype=np.float64)
                self.end_pose[arm_name].append(pose_vec)
            except Exception as e:
                print(f"警告: 采集机械臂 {arm_name} 状态失败: {e}")
    
    def capture_camera_images(self, env, camera_names=None, obs=None):
        """
        采集相机图像数据
        
        Args:
            env: 环境对象
            camera_names: 相机名称列表，默认为所有可用相机
            obs: 观察数据（可选，如果已获取可以直接传入）
        """
        try:
            # 更新渲染
            env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=True)
            
            # 获取观察数据，从中提取相机图像
            if obs is None:
                # 尝试从环境获取观察数据
                try:
                    obs = env.unwrapped.get_observation()
                except:
                    obs = None
            
            if obs is None:
                print("  警告: 无法获取观察数据")
                return
            
            if 'sensor_data' not in obs:
                print("  警告: 观察数据中没有 sensor_data")
                return
            
            sensor_data = obs['sensor_data']
            
            # 获取相机名称列表
            available_cams = []
            for key in sensor_data.keys():
                # 识别相机数据（通常包含 'camera', 'cam', 'rgb' 等关键词）
                if isinstance(sensor_data[key], dict) and 'rgb' in sensor_data[key]:
                    available_cams.append(key)
            
            # 如果未指定相机名称，使用所有可用相机
            if camera_names is None:
                camera_names = available_cams
            
            # 打印可用相机列表（仅首次打印）
            if not hasattr(self, '_cameras_printed'):
                print(f"  可用相机列表: {available_cams}")
                self._cameras_printed = True
            
            # 如果没有可用相机，返回
            if len(camera_names) == 0:
                print("  警告: 没有可用的相机")
                return
            
            for cam_name in camera_names:
                if cam_name in sensor_data:
                    try:
                        # 应用相机名称映射
                        mapped_cam_name = self.CAMERA_NAME_MAPPING.get(cam_name, cam_name)
                        
                        cam_data = sensor_data[cam_name]
                        
                        # ===== 处理 RGB 图像 =====
                        if isinstance(cam_data, dict) and 'rgb' in cam_data:
                            rgb_img = cam_data['rgb']
                        else:
                            rgb_img = cam_data
                        
                        # 处理Tensor类型
                        if hasattr(rgb_img, 'cpu'):
                            rgb_img = rgb_img.cpu().numpy()
                        
                        # 确保图像是uint8格式
                        if rgb_img.max() <= 1.0:
                            rgb_img = (rgb_img * 255).astype(np.uint8)
                        elif rgb_img.dtype != np.uint8:
                            rgb_img = rgb_img.astype(np.uint8)
                        
                        # 确保图像形状正确 (H, W, 3)
                        if len(rgb_img.shape) == 4:
                            rgb_img = rgb_img.squeeze()
                        if len(rgb_img.shape) == 3 and rgb_img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                            rgb_img = np.transpose(rgb_img, (1, 2, 0))
                        
                        # 转换为JPEG字节（与 episode_to_hdf5.py 格式一致）
                        # RGB图像保持uint8格式
                        _, rgb_bytes = cv2.imencode('.jpg', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                        rgb_bytes_data = rgb_bytes.tobytes()
                        
                        # 确保映射后的相机名称在字典中
                        if mapped_cam_name not in self.color_bytes:
                            self.color_bytes[mapped_cam_name] = []
                            self.color_timestamps[mapped_cam_name] = []
                        self.color_bytes[mapped_cam_name].append(rgb_bytes_data)
                        # 记录当前帧的时间戳
                        frame_timestamp = self.timestamps[-1] if self.timestamps else len(self.color_bytes[mapped_cam_name]) / 30.0
                        self.color_timestamps[mapped_cam_name].append(frame_timestamp)
                        
                        # 如果设置了图像保存目录，同时保存到文件夹
                        if hasattr(self, '_image_output_dir') and self._image_output_dir is not None:
                            cam_dir = os.path.join(self._image_output_dir, mapped_cam_name)
                            os.makedirs(cam_dir, exist_ok=True)
                            # 使用时间戳作为文件名（与 episode_to_hdf5.py 格式一致）
                            timestamp = self.timestamps[-1] if self.timestamps else len(self.color_bytes[mapped_cam_name]) / 30.0
                            img_path = os.path.join(cam_dir, f"{timestamp:.9f}.jpg")
                            cv2.imwrite(img_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                        
                        # ===== 处理深度图像 =====
                        if isinstance(cam_data, dict) and 'depth' in cam_data:
                            depth_img = cam_data['depth']
                            
                            # 处理Tensor类型
                            if hasattr(depth_img, 'cpu'):
                                depth_img = depth_img.cpu().numpy()
                            
                            # 确保图像形状正确
                            if len(depth_img.shape) == 4:
                                depth_img = depth_img.squeeze()
                            if len(depth_img.shape) == 3 and depth_img.shape[0] == 1:
                                depth_img = depth_img.squeeze(axis=0)
                            
                            # 转换为PNG字节（与 episode_to_hdf5.py 格式一致）
                            # 深度图像使用uint16格式
                            depth_img = depth_img.astype(np.uint16)
                            _, depth_bytes = cv2.imencode('.png', depth_img)
                            depth_bytes_data = depth_bytes.tobytes()
                            
                            # 确保映射后的相机名称在字典中
                            if mapped_cam_name not in self.depth_bytes:
                                self.depth_bytes[mapped_cam_name] = []
                                self.depth_timestamps[mapped_cam_name] = []
                            self.depth_bytes[mapped_cam_name].append(depth_bytes_data)
                            # 记录当前帧的时间戳
                            frame_timestamp = self.timestamps[-1] if self.timestamps else len(self.depth_bytes[mapped_cam_name]) / 30.0
                            self.depth_timestamps[mapped_cam_name].append(frame_timestamp)
                            
                            # 如果设置了图像保存目录，同时保存深度图到文件夹
                            if hasattr(self, '_image_output_dir') and self._image_output_dir is not None:
                                depth_dir = os.path.join(self._image_output_dir, f"{mapped_cam_name}_depth")
                                os.makedirs(depth_dir, exist_ok=True)
                                # 使用相同的时间戳作为文件名
                                timestamp = self.timestamps[-1] if self.timestamps else len(self.depth_bytes[mapped_cam_name]) / 30.0
                                depth_path = os.path.join(depth_dir, f"{timestamp:.9f}.png")
                                cv2.imwrite(depth_path, depth_img)
                        
                        # 调试信息
                        if not hasattr(self, '_image_printed'):
                            print(f"  相机 {cam_name} -> {mapped_cam_name} RGB形状: {rgb_img.shape}, 数据类型: {rgb_img.dtype}")
                            if isinstance(cam_data, dict) and 'depth' in cam_data:
                                print(f"  相机 {cam_name} -> {mapped_cam_name} 深度图形状: {depth_img.shape}, 数据类型: {depth_img.dtype}")
                            self._image_printed = True
                    except Exception as e:
                        print(f"警告: 采集相机 {cam_name} 图像失败: {e}")
        except Exception as e:
            print(f"警告: 采集相机图像失败: {e}")
    
    def save_to_hdf5_direct(self, h5_path, instruction=""):
        """
        直接保存到指定路径（格式与 episode_to_hdf5.py 完全对齐）
        
        Args:
            h5_path: HDF5文件的完整路径
            instruction: 任务指令文本
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        
        with h5py.File(h5_path, 'w') as h5:
            # 机械臂数据（与episode_to_hdf5.py格式一致）
            arm_group = h5.create_group("arm")
            end_pose_group = arm_group.create_group("endPose")
            effort_group = arm_group.create_group("jointStateEffort")
            position_group = arm_group.create_group("jointStatePosition")
            velocity_group = arm_group.create_group("jointStateVelocity")
            
            for arm in self.ARM_NAMES:
                if self.end_pose[arm]:
                    end_pose_group.create_dataset(arm, data=np.stack(self.end_pose[arm]), compression='gzip')
                if self.position[arm]:
                    position_group.create_dataset(arm, data=np.stack(self.position[arm]), compression='gzip')
                if self.velocity[arm]:
                    velocity_group.create_dataset(arm, data=np.stack(self.velocity[arm]), compression='gzip')
                # 即使没有采集effort数据，也创建空数据集
                effort_group.create_dataset(arm, data=np.array([]), compression='gzip')
            
            # 相机图像数据（使用 vlen uint8 变长字节，与 episode_to_hdf5.py 一致）
            camera_group = h5.create_group("camera")
            color_group = camera_group.create_group("color")
            depth_group = camera_group.create_group("depth")
            
            recorded_cams = set(self.color_bytes.keys()) | set(self.depth_bytes.keys())
            for cam_name in recorded_cams:
                if cam_name in self.color_bytes and self.color_bytes[cam_name]:
                    self._create_bytes_dataset(color_group, cam_name, self.color_bytes[cam_name])
                if cam_name in self.depth_bytes and self.depth_bytes[cam_name]:
                    self._create_bytes_dataset(depth_group, cam_name, self.depth_bytes[cam_name])
            
            # 顶层元数据（与 episode_to_hdf5.py 格式一致）
            h5.create_dataset("instruction", data=instruction)
            h5.create_dataset("timestamp", data=np.asarray(self.timestamps, dtype=np.float64), compression='gzip')
            h5.create_dataset("size", data=np.asarray(len(self.timestamps), dtype=np.int64))
            
            h5.attrs["source_episode_dir"] = str(os.path.dirname(h5_path))
            h5.attrs["alignment_reference"] = "simulation"
            h5.attrs["max_time_delta_sec"] = 0.0
            h5.attrs["end_pose_fields"] = ",".join(self.END_POSE_FIELDS)
    
    def save_to_hdf5(self, output_dir, timestamp=None, instruction=""):
        """
        将采集的数据保存为HDF5文件
        
        Args:
            output_dir: 输出目录
            timestamp: 时间戳（用于生成文件名）
            instruction: 任务指令文本
        
        Returns:
            HDF5文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        h5_path = os.path.join(output_dir, f"trajectory_{timestamp}.h5")
        
        with h5py.File(h5_path, 'w') as h5:
            # 机械臂数据（与episode_to_hdf5.py格式一致）
            arm_group = h5.create_group("arm")
            end_pose_group = arm_group.create_group("endPose")
            position_group = arm_group.create_group("jointStatePosition")
            velocity_group = arm_group.create_group("jointStateVelocity")
            effort_group = arm_group.create_group("jointStateEffort")  # 保留effort文件夹
            
            for arm in self.ARM_NAMES:
                if self.end_pose[arm]:
                    end_pose_group.create_dataset(arm, data=np.stack(self.end_pose[arm]), compression='gzip')
                if self.position[arm]:
                    position_group.create_dataset(arm, data=np.stack(self.position[arm]), compression='gzip')
                if self.velocity[arm]:
                    velocity_group.create_dataset(arm, data=np.stack(self.velocity[arm]), compression='gzip')
                # 即使没有采集effort数据，也创建空数据集
                effort_group.create_dataset(arm, data=np.array([]), compression='gzip')
            
            # 相机图像数据
            camera_group = h5.create_group("camera")
            color_group = camera_group.create_group("color")
            depth_group = camera_group.create_group("depth")
            
            # 使用实际采集到的相机名称
            recorded_cams = set(self.color_bytes.keys()) | set(self.depth_bytes.keys())
            for cam in recorded_cams:
                if cam in self.color_bytes and self.color_bytes[cam]:
                    self._create_bytes_dataset(color_group, cam, self.color_bytes[cam])
                if cam in self.depth_bytes and self.depth_bytes[cam]:
                    self._create_bytes_dataset(depth_group, cam, self.depth_bytes[cam])
            
            # 顶层元数据
            h5.create_dataset("instruction", data=instruction)
            # 时间戳使用普通小数格式保存（避免科学计数法显示问题）
            timestamps_array = np.asarray(self.timestamps, dtype=np.float64)
            h5.create_dataset("timestamp", data=timestamps_array, compression='gzip', dtype='f8')
            h5.create_dataset("size", data=np.asarray(len(self.timestamps), dtype=np.int64))
                
            # 属性信息
            h5.attrs["source_episode_dir"] = str(output_dir)
            h5.attrs["alignment_reference"] = "simulation"
            h5.attrs["max_time_delta_sec"] = 0.0
            h5.attrs["end_pose_fields"] = ",".join(self.END_POSE_FIELDS)
        
        print(f"轨迹数据已保存到 {h5_path}")
        return h5_path
    
    def set_image_output_dir(self, output_dir):
        """
        设置图像保存目录（用于将图片单独保存到文件夹）
        如果之前已经采集了图片，会回溯保存这些图片
        
        Args:
            output_dir: 图像输出目录路径
        """
        self._image_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"  图像将保存到: {output_dir}")
        
        # 回溯保存之前采集的图片
        self._save_pending_images()
    
    def _save_pending_images(self):
        """保存之前采集但尚未保存到文件夹的图片"""
        if not hasattr(self, '_image_output_dir') or self._image_output_dir is None:
            return
        
        # 保存 RGB 图像
        for cam_name, images in self.color_bytes.items():
            cam_dir = os.path.join(self._image_output_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            timestamps = self.color_timestamps.get(cam_name, [])
            for idx, rgb_bytes_data in enumerate(images):
                # 使用时间戳作为文件名
                timestamp = timestamps[idx] if idx < len(timestamps) else idx / 30.0
                img_path = os.path.join(cam_dir, f"{timestamp:.9f}.jpg")
                # 解码 JPEG 字节并保存
                rgb_np = np.frombuffer(rgb_bytes_data, dtype=np.uint8)
                rgb_img = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
                cv2.imwrite(img_path, rgb_img)
        
        # 保存深度图像
        for cam_name, images in self.depth_bytes.items():
            depth_dir = os.path.join(self._image_output_dir, f"{cam_name}_depth")
            os.makedirs(depth_dir, exist_ok=True)
            timestamps = self.depth_timestamps.get(cam_name, [])
            for idx, depth_bytes_data in enumerate(images):
                # 使用时间戳作为文件名
                timestamp = timestamps[idx] if idx < len(timestamps) else idx / 30.0
                depth_path = os.path.join(depth_dir, f"{timestamp:.9f}.png")
                # 解码 PNG 字节并保存
                depth_np = np.frombuffer(depth_bytes_data, dtype=np.uint8)
                depth_img = cv2.imdecode(depth_np, cv2.IMREAD_UNCHANGED)
                cv2.imwrite(depth_path, depth_img)
        
        print(f"  已回溯保存 {len(self.color_bytes)} 个相机的图片")
    
    def _create_bytes_dataset(self, group, name, values):
        """创建变长字节数据集"""
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        dataset = group.create_dataset(name, shape=(len(values),), dtype=dtype)
        dataset[:] = [np.frombuffer(value, dtype=np.uint8) for value in values]


def record_trajectory_and_joints(env, output_dir="trajectory_data", num_steps=100, timestamp=None):
    """
    兼容旧接口的数据采集函数（仅采集机械臂状态）
    
    Args:
        env: 环境对象
        output_dir: 输出目录
        num_steps: 采集步数
        timestamp: 时间戳
    
    Returns:
        HDF5文件路径
    """
    recorder = BioLabDataRecorder()
    agent = env.unwrapped.agent
    
    for step in range(num_steps):
        # 使用零动作让机械臂保持静止（多智能体格式）
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 采集数据
        recorder.capture_robot_state(agent)
        
        if step % 20 == 0:
            print(f"记录步骤 {step}/{num_steps}")
        
        if terminated or truncated:
            break
    
    return recorder.save_to_hdf5(output_dir, timestamp)


def record_with_camera(env, output_dir="trajectory_data", num_steps=100, timestamp=None):
    """
    采集机械臂状态和相机图像（推荐使用）
    
    Args:
        env: 环境对象
        output_dir: 输出目录
        num_steps: 采集步数
        timestamp: 时间戳
    
    Returns:
        HDF5文件路径
    """
    recorder = BioLabDataRecorder()
    agent = env.unwrapped.agent
    
    for step in range(num_steps):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 采集机械臂状态和相机图像
        recorder.capture_robot_state(agent)
        recorder.capture_camera_images(env)
        
        if step % 20 == 0:
            print(f"记录步骤 {step}/{num_steps}")
    
    return recorder.save_to_hdf5(output_dir, timestamp)