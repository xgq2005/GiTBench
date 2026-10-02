from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.tasks.gitbench_paths import project_asset_path
from mani_skill.utils.io_utils import load_json

CHE_DATASET = dict()


def _load_industry_dataset():
    global CHE_DATASET
    CHE_DATASET = {
        "model_data": load_json(project_asset_path("industry", "info_raw.json")),
    }


def get_industry_builder(
    scene: ManiSkillScene, id: str, add_collision: bool = True, add_visual: bool = True
):
    if "industry" not in CHE_DATASET:
        _load_industry_dataset()

    model_db = CHE_DATASET["model_data"]

    builder = scene.create_actor_builder()

    metadata = model_db[id]
    density = metadata.get("density", 1000)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]
    physical_material = None
    model_dir = project_asset_path("industry", "models", id)
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )
    if add_visual:
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    return builder
