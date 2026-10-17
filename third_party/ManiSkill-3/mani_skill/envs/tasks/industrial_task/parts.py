import numpy as np
import sapien
from mani_skill.utils.building import actors


def generate_random_color():
    """
    生成随机颜色
    
    Returns:
        随机颜色 [r, g, b, a]，其中alpha固定为1
    """
    return [np.random.rand(), np.random.rand(), np.random.rand(), 1]


def build_cube_part(scene, half_size=0.025, color=None, name="cube_part", initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, half_size])
    if color is None:
        color = generate_random_color()

    return actors.build_cube(
        scene,
        half_size=half_size,
        color=color,
        name=name,
        initial_pose=initial_pose,
    )


def build_sphere_part(scene, radius=0.025, color=None, name="sphere_part", initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, radius])
    if color is None:
        color = generate_random_color()

    return actors.build_sphere(
        scene,
        radius=radius,
        color=color,
        name=name,
        initial_pose=initial_pose,
    )


def build_cylinder_part(scene, radius=0.025, half_length=0.04, color=None, name="cylinder_part",
                        initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, half_length])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    q = [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0]
    pose = sapien.Pose(p=[0, 0, 0], q=q)
    
    builder.add_cylinder_collision(radius=radius, half_length=half_length, material=material, pose=pose)
    builder.add_cylinder_visual(radius=radius, half_length=half_length, material=render_material, pose=pose)
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_l_shape_part(scene, side_length=0.08, thickness=0.025, color=None, name="l_shape_part",
                       initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, side_length / 2])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    half_thickness = thickness / 2
    half_height = side_length / 2
    
    horizontal_half_size = [half_height-half_thickness, half_thickness, half_thickness]
    horizontal_pose = sapien.Pose(p=[half_height / 2, 0, half_height - 0.02])
    
    vertical_half_size = [half_thickness, half_thickness, half_height-half_thickness]
    vertical_pose = sapien.Pose(p=[0, 0, half_height / 2 - 0.02])
    
    builder.add_box_collision(half_size=horizontal_half_size, material=material, pose=horizontal_pose)
    builder.add_box_visual(half_size=horizontal_half_size, material=render_material, pose=horizontal_pose)
    
    builder.add_box_collision(half_size=vertical_half_size, material=material, pose=vertical_pose)
    builder.add_box_visual(half_size=vertical_half_size, material=render_material, pose=vertical_pose)
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_t_shape_part(scene, side_length=0.06, thickness=0.025, color=None, name="t_shape_part",
                       initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, side_length / 2])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    half_thickness = thickness / 2
    half_height = side_length / 2
    
    horizontal_half_size = [half_height, half_thickness, half_thickness]
    horizontal_pose = sapien.Pose(p=[0, 0, side_length - half_thickness - 0.03])
    
    vertical_half_size = [half_thickness, half_thickness, half_height]
    vertical_pose = sapien.Pose(p=[0, 0, half_height / 2 - 0.03])
    
    builder.add_box_collision(half_size=horizontal_half_size, material=material, pose=horizontal_pose)
    builder.add_box_visual(half_size=horizontal_half_size, material=render_material, pose=horizontal_pose)
    
    builder.add_box_collision(half_size=vertical_half_size, material=material, pose=vertical_pose)
    builder.add_box_visual(half_size=vertical_half_size, material=render_material, pose=vertical_pose)
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_h_shape_part(scene, side_length=0.08, thickness=0.025, color=None, name="h_shape_part",
                       initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, side_length / 2])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    half_thickness = thickness / 2
    half_height = side_length / 2
    
    left_vertical_half_size = [half_thickness, half_thickness, half_height-half_thickness]
    left_vertical_pose = sapien.Pose(p=[-half_height / 2, 0, half_height / 2 - 0.03])
    
    right_vertical_half_size = [half_thickness, half_thickness, half_height-half_thickness]
    right_vertical_pose = sapien.Pose(p=[half_height / 2, 0, half_height / 2 - 0.03])
    
    horizontal_half_size = [half_height-half_thickness, half_thickness, half_thickness]
    horizontal_pose = sapien.Pose(p=[0, 0, half_height/2 - 0.03])
    
    builder.add_box_collision(half_size=left_vertical_half_size, material=material, pose=left_vertical_pose)
    builder.add_box_visual(half_size=left_vertical_half_size, material=render_material, pose=left_vertical_pose)
    
    builder.add_box_collision(half_size=right_vertical_half_size, material=material, pose=right_vertical_pose)
    builder.add_box_visual(half_size=right_vertical_half_size, material=render_material, pose=right_vertical_pose)
    
    builder.add_box_collision(half_size=horizontal_half_size, material=material, pose=horizontal_pose)
    builder.add_box_visual(half_size=horizontal_half_size, material=render_material, pose=horizontal_pose)
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_ring_part(scene, inner_radius=0.01, outer_radius=0.025, height=0.03,
                    color=None, name="ring_part", initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, height / 2])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    half_height = height / 2
    thickness = outer_radius - inner_radius
    half_thickness = thickness / 2
    avg_radius = (inner_radius + outer_radius) / 2
    
    num_segments = 8
    segment_angle = 2 * np.pi / num_segments
    segment_length = avg_radius * segment_angle
    half_length = segment_length / 2
    
    box_half_size = [half_length, half_thickness, half_height]
    
    for i in range(num_segments):
        angle = i * segment_angle
        x = avg_radius * np.cos(angle)
        y = avg_radius * np.sin(angle)
        q = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        pose = sapien.Pose(p=[x, y, 0], q=q)
        builder.add_box_collision(half_size=box_half_size, material=material, pose=pose)
        builder.add_box_visual(half_size=box_half_size, material=render_material, pose=pose)
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_box_part(scene, half_sizes=[0.0125, 0.0125, 0.04], color=None, name="box_part", initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, half_sizes[2]])
    if color is None:
        color = generate_random_color()

    return actors.build_box(
        scene,
        half_sizes=half_sizes,
        color=color,
        name=name,
        initial_pose=initial_pose,
    )


def build_hexagonal_prism_part(scene, side_length=0.025, height=0.08, color=None, name="hex_prism_part",
                               initial_pose=None):
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, height / 2])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    radius = side_length
    half_height = height / 2
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    box_width = radius * np.sqrt(3)
    box_half_size = [box_width / 2, radius / 2, half_height]
    
    for i in range(3):
        angle = i * np.pi / 3
        x = 0
        y = 0
        q = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        pose = sapien.Pose(p=[x, y, 0], q=q)
        builder.add_box_collision(half_size=box_half_size, material=material, pose=pose)
        builder.add_box_visual(half_size=box_half_size, material=render_material, pose=pose)
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def build_screw_part(scene, head_radius=0.025, head_height=0.02, shaft_radius=0.01, shaft_height=0.06,
                     color=None, name="screw_part", initial_pose=None):
    if initial_pose is None:
        total_height = head_height + shaft_height
        initial_pose = sapien.Pose(p=[0, 0, total_height])
    if color is None:
        color = generate_random_color()

    builder = scene.create_actor_builder()
    
    material = sapien.pysapien.physx.PhysxMaterial(
        static_friction=5.0,
        dynamic_friction=3.0,
        restitution=0.1
    )
    
    render_material = sapien.render.RenderMaterial(base_color=color)
    
    head_half_height = head_height / 2
    box_width = head_radius * np.sqrt(3)
    box_half_size = [box_width / 2, head_radius / 2, head_half_height]
    
    head_z_offset = shaft_height + head_half_height
    
    for i in range(3):
        angle = i * np.pi / 3
        x = 0
        y = 0
        q = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        pose = sapien.Pose(p=[x, y, head_z_offset-0.12], q=q)
        builder.add_box_collision(half_size=box_half_size, material=material, pose=pose)
        builder.add_box_visual(half_size=box_half_size, material=render_material, pose=pose)
    
    shaft_half_length = shaft_height / 2
    shaft_z_offset = shaft_height
    builder.add_cylinder_collision(radius=shaft_radius, half_length=shaft_half_length, material=material,
                                   pose=sapien.Pose(p=[0, 0, 2*head_height+shaft_z_offset-0.12],q=[0,0.707,0,0.707]))
    builder.add_cylinder_visual(radius=shaft_radius, half_length=shaft_half_length, material=render_material,
                                pose=sapien.Pose(p=[0, 0, 2*head_height+shaft_z_offset-0.12],q=[0,0.707,0,0.707]))
    
    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    
    return actor


def create_part_by_type(scene, part_type, name=None, color=None, initial_pose=None):
    if name is None:
        name = f"{part_type}_part"

    part_creators = {
        "cube": build_cube_part,
        "sphere": build_sphere_part,
        "cylinder": build_cylinder_part,
        "box": build_box_part,
        "hex_prism": build_hexagonal_prism_part,
        "screw": build_screw_part,
        "ring": build_ring_part,
        "l_shape": build_l_shape_part,
        "t_shape": build_t_shape_part,
        "h_shape": build_h_shape_part,
    }

    if part_type not in part_creators:
        raise ValueError(f"Unknown part type: {part_type}. Available types: {list(part_creators.keys())}")

    creator = part_creators[part_type]
    return creator(scene, color=color, name=name, initial_pose=initial_pose)


def get_part_dimensions(part_type):
    dimensions = {
        "cube": {
            "type": "正方体",
            "size": [0.05, 0.05, 0.05],
            "description": "边长5cm的正方体"
        },
        "sphere": {
            "type": "球体",
            "radius": 0.025,
            "diameter": 0.05,
            "description": "半径2.5cm的球体"
        },
        "cylinder": {
            "type": "圆柱体",
            "radius": 0.025,
            "height": 0.08,
            "description": "半径2.5cm，高8cm的圆柱体"
        },
        "box": {
            "type": "长方体",
            "size": [0.025, 0.025, 0.08],
            "description": "2.5x2.5x8cm的长方体"
        },
        "hex_prism": {
            "type": "六角柱体",
            "side_length": 0.025,
            "height": 0.08,
            "description": "六边形边长2.5cm，高8cm的六角柱体"
        },
        "screw": {
            "type": "螺丝",
            "head_radius": 0.025,
            "head_height": 0.02,
            "shaft_radius": 0.01,
            "shaft_height": 0.06,
            "total_height": 0.08,
            "description": "六角柱头部(半径2.5cm,高2cm)和圆柱杆部(半径1cm,高6cm)组成的螺丝"
        },
        "ring": {
            "type": "圆环",
            "inner_radius": 0.005,
            "outer_radius": 0.0125,
            "height": 0.03,
            "description": "内径1cm,外径2.5cm,高3cm的圆环"
        },
        "l_shape": {
            "type": "L形零件",
            "side_length": 0.05,
            "thickness": 0.005,
            "description": "每边长5cm，厚度0.5cm的L形零件"
        },
        "t_shape": {
            "type": "T形零件",
            "side_length": 0.05,
            "thickness": 0.005,
            "description": "每边长5cm，厚度0.5cm的T形零件"
        },
        "h_shape": {
            "type": "H形零件",
            "side_length": 0.05,
            "thickness": 0.005,
            "description": "每边长5cm，厚度0.5cm的H形零件"
        }
    }

    if part_type not in dimensions:
        raise ValueError(f"Unknown part type: {part_type}. Available types: {list(dimensions.keys())}")

    return dimensions[part_type]