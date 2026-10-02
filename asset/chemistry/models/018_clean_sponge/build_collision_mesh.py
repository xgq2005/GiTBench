import coacd
import trimesh
import numpy as np

def merge_meshes(mesh_list):
    """将多个网格合并为一个网格"""
    if not mesh_list:
        return None
    
    vertices_list = []
    faces_list = []
    face_offset = 0
    
    for mesh in mesh_list:
        vertices_list.append(mesh.vertices)
        faces_list.append(mesh.faces + face_offset)
        face_offset += len(mesh.vertices)
    
    merged_vertices = np.vstack(vertices_list)
    merged_faces = np.vstack(faces_list)
    
    return trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces)

def main():
    # 1. 加载输入网格
    input_path = "./textured.obj"
    try:
        mesh = trimesh.load(input_path, force='mesh')
    except Exception as e:
        print(f"无法加载网格文件 {input_path}: {e}")
        return

    # 确保是三角网格
    if not mesh.is_empty and (len(mesh.faces) == 0 or mesh.faces.shape[1] != 3):
        print("输入网格不是三角网格，正在尝试三角化...")
        mesh = mesh.triangulate()

    # 转换为CoACD所需的Mesh格式
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    # 2. 运行近似凸分解
    print("正在运行CoACD进行凸分解...")
    try:
        # 尝试使用不同的参数组合
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=0.02,     # 减小阈值增加凸包数量
            max_convex_hull=60,  # 限制最大凸包数量
            mcts_iterations=40,  # 增加迭代次数
            resolution=4096      # 提高分辨率
        )
    except Exception as e:
        print(f"参数化调用失败: {e}")
        print("尝试使用默认参数...")
        try:
            parts = coacd.run_coacd(coacd_mesh)
        except Exception as e2:
            print(f"CoACD处理失败: {e2}")
            return

    print(f"凸分解完成，得到 {len(parts)} 个凸包")
    
    # 3. 处理输出
    trimesh_parts = []
    
    for i, part in enumerate(parts):
        try:
            vertices = np.array(part[0], dtype=np.float32)
            faces = np.array(part[1], dtype=np.int32)
            
            if len(faces) > 0:
                trimesh_parts.append(trimesh.Trimesh(vertices=vertices, faces=faces))
                print(f"  凸包{i}: {len(vertices)}顶点, {len(faces)}面")
                
        except Exception as e:
            print(f"处理第 {i} 个凸包时出错: {e}")
            continue

    if len(trimesh_parts) == 0:
        print("未能成功创建任何trimesh对象")
        return

    # 4. 合并所有凸包为一个网格
    print(f"成功创建了 {len(trimesh_parts)} 个trimesh对象，正在合并...")
    combined_mesh = merge_meshes(trimesh_parts)
    
    if combined_mesh is None:
        print("无法合并网格")
        return

    # 5. 保存为单个PLY文件
    output_path = "./collision.ply"
    try:
        combined_mesh.export(output_path, file_type='ply')
        print(f"\n✅ 合并后的碰撞网格已保存至: {output_path}")
        print(f"  顶点数: {len(combined_mesh.vertices)}")
        print(f"  面数: {len(combined_mesh.faces)}")
        print(f"  由 {len(trimesh_parts)} 个凸包合并而成")
        
    except Exception as e:
        print(f"保存PLY文件失败: {e}")

if __name__ == "__main__":
    main()
