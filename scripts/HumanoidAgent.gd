extends CharacterBody3D
class_name HumanoidAgent

const ACTION_SIZE := 8
const OBS_SIZE := 15

@export var move_speed: float = 7.5
@export var move_accel: float = 18.0
@export var gravity: float = 18.0
@export var jump_velocity: float = 6.2
@export var max_fall_speed: float = 32.0
@export var body_turn_speed: float = 8.0
@export var model_scene_path: String = "res://Models/Mimic/mimic.tscn"
@export var model_scale: Vector3 = Vector3.ONE
@export var model_base_offset: Vector3 = Vector3.ZERO
@export var use_imported_model: bool = true
@export var hide_fallback_rig_when_model_loaded: bool = true

const MODEL_CANDIDATE_PATHS := [
	"res://Models/Mimic/mimic.tscn",
	"res://Models/RobotKyle.fbx",
	"res://models/RobotKyle.fbx",
	"res://Models/RobotKyle.glb",
	"res://models/RobotKyle.glb"
]

const BONE_GROUPS := {
	"head": ["head", "Head", "mixamorig:Head", "neck"],
	"left_arm": ["upperarm_l", "UpperArm_L", "LeftArm", "mixamorig:LeftArm", "leftarm", "arm_l", "shoulder_l"],
	"right_arm": ["upperarm_r", "UpperArm_R", "RightArm", "mixamorig:RightArm", "rightarm", "arm_r", "shoulder_r"],
	"left_leg": ["thigh_l", "LeftUpLeg", "mixamorig:LeftUpLeg", "leftupleg", "leg_l", "calf_l", "shin_l"],
	"right_leg": ["thigh_r", "RightUpLeg", "mixamorig:RightUpLeg", "rightupleg", "leg_r", "calf_r", "shin_r"]
}

const MIMIC_TEXTURE_PATHS := {
	"albedo": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_BaseColor.png",
	"normal": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_Normal.png",
	"roughness": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_Roughness.png",
	"metallic": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_Metallic.png",
	"ao": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_Occlusion.png",
	"emission": "res://Models/Mimic/textures/TEX_Old_Endo_Main_00_Emissive.png"
}

@onready var visual_root: Node3D = $VisualRoot
@onready var model_container: Node3D = $VisualRoot/ModelContainer
@onready var head_pivot: Node3D = $VisualRoot/HeadPivot
@onready var left_arm_pivot: Node3D = $VisualRoot/LeftArmPivot
@onready var right_arm_pivot: Node3D = $VisualRoot/RightArmPivot
@onready var left_leg_pivot: Node3D = $VisualRoot/LeftLegPivot
@onready var right_leg_pivot: Node3D = $VisualRoot/RightLegPivot
@onready var torso_mesh: MeshInstance3D = $VisualRoot/Torso
@onready var head_mesh: MeshInstance3D = $VisualRoot/HeadPivot/Head
@onready var left_arm_mesh: MeshInstance3D = $VisualRoot/LeftArmPivot/LeftArm
@onready var right_arm_mesh: MeshInstance3D = $VisualRoot/RightArmPivot/RightArm
@onready var left_leg_mesh: MeshInstance3D = $VisualRoot/LeftLegPivot/LeftLeg
@onready var right_leg_mesh: MeshInstance3D = $VisualRoot/RightLegPivot/RightLeg
@onready var torso_collision: CollisionShape3D = $CollisionShape3D
@onready var head_collision: CollisionShape3D = $HeadCollision
@onready var left_leg_collision: CollisionShape3D = $LeftLegCollision
@onready var right_leg_collision: CollisionShape3D = $RightLegCollision

var survived_time: float = 0.0
var _pending_action: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
var _model_root: Node3D = null
var _skeleton: Skeleton3D = null
var _ragdoll_simulator: PhysicalBoneSimulator3D = null
var _bone_indices: Dictionary = {
	"head": -1,
	"left_arm": -1,
	"right_arm": -1,
	"left_leg": -1,
	"right_leg": -1
}
var _rest_bone_rotations: Dictionary = {}
var _ragdoll_active: bool = false


func _ready() -> void:
	_spawn_imported_model()
	_set_neutral_pose()


func _physics_process(delta: float) -> void:
	survived_time += delta
	if _ragdoll_active:
		_update_segmented_colliders(delta)
		move_and_slide()
		return

	_apply_action_to_motion(delta)
	_animate_body_parts(delta)
	_update_segmented_colliders(delta)
	move_and_slide()


func reset_agent(spawn_position: Vector3) -> void:
	global_position = spawn_position
	velocity = Vector3.ZERO
	survived_time = 0.0
	_pending_action = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	visual_root.rotation = Vector3.ZERO
	set_ragdoll_enabled(false)
	_set_neutral_pose()


func apply_action(action: PackedFloat32Array) -> void:
	if action.size() < ACTION_SIZE:
		return

	_pending_action = PackedFloat32Array(action)


func get_observation(pursuer_position: Vector3, pursuer_velocity: Vector3, elapsed: float, episode_limit: float) -> PackedFloat32Array:
	var delta: Vector3 = pursuer_position - global_position
	var distance: float = delta.length()

	var forward: Vector3 = -visual_root.global_basis.z
	var forward_planar := Vector2(forward.x, forward.z)
	var threat_planar := Vector2(delta.x, delta.z)
	var angle_to_threat: float = 0.0
	if forward_planar.length() > 0.001 and threat_planar.length() > 0.001:
		angle_to_threat = forward_planar.normalized().angle_to(threat_planar.normalized())

	return PackedFloat32Array([
		delta.x,
		delta.y,
		delta.z,
		velocity.x,
		velocity.y,
		velocity.z,
		pursuer_velocity.x,
		pursuer_velocity.y,
		pursuer_velocity.z,
		distance,
		1.0 if is_on_floor() else 0.0,
		clamp(elapsed / max(episode_limit, 0.001), 0.0, 1.0),
		angle_to_threat,
		Vector2(velocity.x, velocity.z).length(),
		Vector2(pursuer_velocity.x, pursuer_velocity.z).length()
	])


func get_agent_status(distance_to_pursuer: float) -> Dictionary:
	return {
		"survival_time": survived_time,
		"height": global_position.y,
		"speed": Vector2(velocity.x, velocity.z).length(),
		"vertical_speed": velocity.y,
		"on_floor": is_on_floor(),
		"distance": distance_to_pursuer,
		"action_jump": _pending_action[2]
	}


func get_last_action() -> PackedFloat32Array:
	return PackedFloat32Array(_pending_action)


func _apply_action_to_motion(delta: float) -> void:
	var move_input := Vector2(_pending_action[0], _pending_action[1]).clamp(Vector2(-1.0, -1.0), Vector2(1.0, 1.0))
	var desired_velocity := Vector3(move_input.x, 0.0, move_input.y) * move_speed

	velocity.x = move_toward(velocity.x, desired_velocity.x, move_accel * delta)
	velocity.z = move_toward(velocity.z, desired_velocity.z, move_accel * delta)

	if not is_on_floor():
		velocity.y = max(velocity.y - gravity * delta, -max_fall_speed)
	elif _pending_action[2] > 0.35:
		velocity.y = jump_velocity

	var planar_velocity := Vector2(velocity.x, velocity.z)
	if planar_velocity.length() > 0.15:
		var target_yaw := atan2(planar_velocity.x, planar_velocity.y)
		visual_root.rotation.y = lerp_angle(visual_root.rotation.y, target_yaw, body_turn_speed * delta)


func _animate_body_parts(delta: float) -> void:
	var walk_amount: float = clampf(Vector2(velocity.x, velocity.z).length() / max(move_speed, 0.001), 0.0, 1.0)
	var gait: float = sin(survived_time * 7.0) * walk_amount

	var head_target: float = clampf(_pending_action[3], -1.0, 1.0) * 0.55
	var left_arm_target: float = clampf(_pending_action[4], -1.0, 1.0) * 0.9 + gait * 0.65
	var right_arm_target: float = clampf(_pending_action[5], -1.0, 1.0) * 0.9 - gait * 0.65
	var left_leg_target: float = clampf(_pending_action[6], -1.0, 1.0) * 0.8 - gait * 0.95
	var right_leg_target: float = clampf(_pending_action[7], -1.0, 1.0) * 0.8 + gait * 0.95

	head_pivot.rotation.x = lerp(head_pivot.rotation.x, head_target, delta * 9.0)
	left_arm_pivot.rotation.x = lerp(left_arm_pivot.rotation.x, left_arm_target, delta * 10.0)
	right_arm_pivot.rotation.x = lerp(right_arm_pivot.rotation.x, right_arm_target, delta * 10.0)
	left_leg_pivot.rotation.x = lerp(left_leg_pivot.rotation.x, left_leg_target, delta * 10.0)
	right_leg_pivot.rotation.x = lerp(right_leg_pivot.rotation.x, right_leg_target, delta * 10.0)

	_apply_bone_pose("head", head_target, delta * 9.0)
	_apply_bone_pose("left_arm", left_arm_target, delta * 10.0)
	_apply_bone_pose("right_arm", right_arm_target, delta * 10.0)
	_apply_bone_pose("left_leg", left_leg_target, delta * 10.0)
	_apply_bone_pose("right_leg", right_leg_target, delta * 10.0)


func _spawn_imported_model() -> void:
	if not use_imported_model:
		return

	var resolved_model_path := _resolve_model_scene_path()
	if resolved_model_path != "":
		model_scene_path = resolved_model_path

	var resource := load(model_scene_path)
	if resource == null:
		push_warning("Nao foi possivel carregar modelo 3D em: %s" % model_scene_path)
		return

	if resource is PackedScene:
		var model_instance := (resource as PackedScene).instantiate()
		model_instance.name = "MimicModel"
		if model_instance is Node3D:
			var model_instance_3d := model_instance as Node3D
			model_instance_3d.position = model_base_offset
			model_instance_3d.scale = model_scale
			_model_root = model_instance_3d
		model_container.add_child(model_instance)
		_disable_model_animations(_model_root)
		_auto_fit_model_to_ground()
		_apply_mimic_textures()
		_set_fallback_rig_visible(not hide_fallback_rig_when_model_loaded)
		_cache_skeleton_and_ragdoll_nodes()
		_map_body_bones()
		return

	if resource is Mesh:
		var mesh_instance := MeshInstance3D.new()
		mesh_instance.name = "MimicModel"
		mesh_instance.mesh = resource as Mesh
		mesh_instance.position = model_base_offset
		mesh_instance.scale = model_scale
		model_container.add_child(mesh_instance)
		_set_fallback_rig_visible(not hide_fallback_rig_when_model_loaded)


func _resolve_model_scene_path() -> String:
	if ResourceLoader.exists(model_scene_path):
		return model_scene_path

	for path in MODEL_CANDIDATE_PATHS:
		if ResourceLoader.exists(path):
			return path

	return ""


func _set_neutral_pose() -> void:
	head_pivot.rotation = Vector3.ZERO
	left_arm_pivot.rotation = Vector3.ZERO
	right_arm_pivot.rotation = Vector3.ZERO
	left_leg_pivot.rotation = Vector3.ZERO
	right_leg_pivot.rotation = Vector3.ZERO


func _set_fallback_rig_visible(visible: bool) -> void:
	torso_mesh.visible = visible
	head_mesh.visible = visible
	left_arm_mesh.visible = visible
	right_arm_mesh.visible = visible
	left_leg_mesh.visible = visible
	right_leg_mesh.visible = visible


func set_ragdoll_enabled(enabled: bool) -> void:
	_ragdoll_active = enabled
	if _ragdoll_simulator != null:
		_ragdoll_simulator.active = enabled
	elif _skeleton != null:
		if enabled:
			_skeleton.physical_bones_start_simulation()
		else:
			_skeleton.physical_bones_stop_simulation()


func _cache_skeleton_and_ragdoll_nodes() -> void:
	_skeleton = _find_first_node_of_type(_model_root, "Skeleton3D") as Skeleton3D
	_ragdoll_simulator = _find_first_node_of_type(_model_root, "PhysicalBoneSimulator3D") as PhysicalBoneSimulator3D


func _map_body_bones() -> void:
	if _skeleton == null:
		push_warning("Modelo Mimic nao possui Skeleton3D acessivel para mapear membros")
		return

	_rest_bone_rotations.clear()

	for key in BONE_GROUPS.keys():
		var aliases: Array = BONE_GROUPS[key]
		var bone_index: int = _find_bone_index_fuzzy(aliases)
		_bone_indices[key] = bone_index
		if bone_index >= 0:
			_rest_bone_rotations[key] = _skeleton.get_bone_pose_rotation(bone_index)


func _apply_bone_pose(group_name: String, x_rotation: float, weight: float) -> void:
	if _ragdoll_active:
		return
	if _skeleton == null:
		return

	var bone_idx: int = int(_bone_indices.get(group_name, -1))
	if bone_idx < 0:
		return

	var current: Quaternion = _skeleton.get_bone_pose_rotation(bone_idx)
	var rest_rotation: Quaternion = _rest_bone_rotations.get(group_name, Quaternion.IDENTITY) as Quaternion
	var target: Quaternion = (rest_rotation as Quaternion) * Quaternion.from_euler(Vector3(x_rotation, 0.0, 0.0))
	var t: float = clampf(weight, 0.0, 1.0)
	_skeleton.set_bone_pose_rotation(bone_idx, current.slerp(target, t))


func _find_first_node_of_type(root: Node, class_name_hint: String) -> Node:
	if root == null:
		return null
	if root.is_class(class_name_hint):
		return root

	for child in root.get_children():
		var child_node := child as Node
		if child_node == null:
			continue
		var found := _find_first_node_of_type(child_node, class_name_hint)
		if found != null:
			return found

	return null


func _auto_fit_model_to_ground() -> void:
	if _model_root == null:
		return

	var mesh_nodes := _collect_mesh_instances(_model_root)
	if mesh_nodes.is_empty():
		return

	var lowest_y: float = INF
	for mesh_node in mesh_nodes:
		var mi := mesh_node as MeshInstance3D
		if mi == null or mi.mesh == null:
			continue
		var local_aabb: AABB = mi.mesh.get_aabb()
		var corners := [
			local_aabb.position,
			local_aabb.position + Vector3(local_aabb.size.x, 0.0, 0.0),
			local_aabb.position + Vector3(0.0, local_aabb.size.y, 0.0),
			local_aabb.position + Vector3(0.0, 0.0, local_aabb.size.z),
			local_aabb.position + Vector3(local_aabb.size.x, local_aabb.size.y, 0.0),
			local_aabb.position + Vector3(local_aabb.size.x, 0.0, local_aabb.size.z),
			local_aabb.position + Vector3(0.0, local_aabb.size.y, local_aabb.size.z),
			local_aabb.position + local_aabb.size
		]
		for corner in corners:
			var global_corner := mi.to_global(corner)
			if global_corner.y < lowest_y:
				lowest_y = global_corner.y

	if lowest_y == INF:
		return

	_model_root.position.y += (0.03 - lowest_y)


func _collect_mesh_instances(root: Node) -> Array:
	var out: Array = []
	if root == null:
		return out

	if root is MeshInstance3D:
		out.append(root)

	for child in root.get_children():
		var child_node := child as Node
		if child_node == null:
			continue
		out.append_array(_collect_mesh_instances(child_node))

	return out


func _update_segmented_colliders(delta: float) -> void:
	# Multi-part hitbox approximation keeps feet/head out of the floor while preserving CharacterBody stability.
	if torso_collision != null:
		torso_collision.position = torso_collision.position.lerp(Vector3(0.0, 1.0, 0.0), clampf(delta * 8.0, 0.0, 1.0))

	var head_target := Vector3(0.0, 1.9, 0.0)
	if head_collision != null:
		head_collision.position = head_collision.position.lerp(head_target, clampf(delta * 9.0, 0.0, 1.0))

	var left_leg_target := Vector3(-0.18, 0.45 + left_leg_pivot.rotation.x * 0.08, 0.0)
	if left_leg_collision != null:
		left_leg_collision.position = left_leg_collision.position.lerp(left_leg_target, clampf(delta * 10.0, 0.0, 1.0))

	var right_leg_target := Vector3(0.18, 0.45 + right_leg_pivot.rotation.x * 0.08, 0.0)
	if right_leg_collision != null:
		right_leg_collision.position = right_leg_collision.position.lerp(right_leg_target, clampf(delta * 10.0, 0.0, 1.0))


func _find_bone_index_fuzzy(aliases: Array) -> int:
	if _skeleton == null:
		return -1

	for alias in aliases:
		var exact_index: int = _skeleton.find_bone(StringName(alias))
		if exact_index >= 0:
			return exact_index

	var normalized_aliases: Array[String] = []
	for alias in aliases:
		normalized_aliases.append(_normalize_bone_name(str(alias)))

	var best_idx: int = -1
	var best_score: int = -1
	for i in range(_skeleton.get_bone_count()):
		var bone_name := _normalize_bone_name(_skeleton.get_bone_name(i))
		var score: int = 0
		for alias in normalized_aliases:
			if alias.is_empty():
				continue
			if bone_name.find(alias) >= 0 or alias.find(bone_name) >= 0:
				score = max(score, alias.length())
			if alias.contains("left") and (bone_name.contains("left") or bone_name.ends_with("l")):
				score += 2
			if alias.contains("right") and (bone_name.contains("right") or bone_name.ends_with("r")):
				score += 2

		if score > best_score:
			best_score = score
			best_idx = i

	if best_score < 3:
		return -1
	return best_idx


func _normalize_bone_name(name: String) -> String:
	return name.to_lower().replace("mixamorig:", "").replace("_", "").replace("-", "").strip_edges()


func _disable_model_animations(root: Node) -> void:
	if root == null:
		return

	if root is AnimationPlayer:
		var anim_player := root as AnimationPlayer
		anim_player.stop()
		anim_player.active = false

	if root.is_class("AnimationTree"):
		root.set("active", false)

	if root.is_class("AnimationMixer"):
		root.set("active", false)

	for child in root.get_children():
		var child_node := child as Node
		if child_node != null:
			_disable_model_animations(child_node)


func _apply_mimic_textures() -> void:
	if _model_root == null:
		return

	var albedo := load(str(MIMIC_TEXTURE_PATHS["albedo"])) as Texture2D
	if albedo == null:
		return

	var normal := load(str(MIMIC_TEXTURE_PATHS["normal"])) as Texture2D
	var roughness := load(str(MIMIC_TEXTURE_PATHS["roughness"])) as Texture2D
	var metallic := load(str(MIMIC_TEXTURE_PATHS["metallic"])) as Texture2D
	var ao := load(str(MIMIC_TEXTURE_PATHS["ao"])) as Texture2D
	var emission := load(str(MIMIC_TEXTURE_PATHS["emission"])) as Texture2D

	var mat := StandardMaterial3D.new()
	mat.albedo_texture = albedo
	mat.roughness_texture = roughness
	mat.metallic_texture = metallic
	mat.normal_enabled = normal != null
	mat.normal_texture = normal
	mat.ao_enabled = ao != null
	mat.ao_texture = ao
	mat.emission_enabled = emission != null
	mat.emission_texture = emission
	mat.emission_energy_multiplier = 1.5
	mat.roughness = 1.0
	mat.metallic = 1.0

	for mesh_node in _collect_mesh_instances(_model_root):
		var mi := mesh_node as MeshInstance3D
		if mi == null or mi.mesh == null:
			continue
		var surface_count: int = mi.mesh.get_surface_count()
		for i in range(surface_count):
			mi.set_surface_override_material(i, mat)
