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
@export var model_scene_path: String = "res://Models/RobotKyle.fbx"
@export var model_scale: Vector3 = Vector3.ONE * 0.015
@export var use_imported_model: bool = true

const MODEL_CANDIDATE_PATHS := [
	"res://Models/RobotKyle.fbx",
	"res://models/RobotKyle.fbx",
	"res://Models/RobotKyle.glb",
    "res://models/RobotKyle.glb"
]

@onready var visual_root: Node3D = $VisualRoot
@onready var model_container: Node3D = $VisualRoot/ModelContainer
@onready var head_pivot: Node3D = $VisualRoot/HeadPivot
@onready var left_arm_pivot: Node3D = $VisualRoot/LeftArmPivot
@onready var right_arm_pivot: Node3D = $VisualRoot/RightArmPivot
@onready var left_leg_pivot: Node3D = $VisualRoot/LeftLegPivot
@onready var right_leg_pivot: Node3D = $VisualRoot/RightLegPivot

var survived_time: float = 0.0
var _pending_action: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


func _ready() -> void:
	_spawn_imported_model()
	_set_neutral_pose()


func _physics_process(delta: float) -> void:
	survived_time += delta
	_apply_action_to_motion(delta)
	_animate_body_parts(delta)
	move_and_slide()


func reset_agent(spawn_position: Vector3) -> void:
	global_position = spawn_position
	velocity = Vector3.ZERO
	survived_time = 0.0
	_pending_action = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	visual_root.rotation = Vector3.ZERO
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
	var left_arm_target: float = clampf(_pending_action[4], -1.0, 1.0) * 0.9 + gait * 0.35
	var right_arm_target: float = clampf(_pending_action[5], -1.0, 1.0) * 0.9 - gait * 0.35
	var left_leg_target: float = clampf(_pending_action[6], -1.0, 1.0) * 0.8 - gait * 0.5
	var right_leg_target: float = clampf(_pending_action[7], -1.0, 1.0) * 0.8 + gait * 0.5

	head_pivot.rotation.x = lerp(head_pivot.rotation.x, head_target, delta * 9.0)
	left_arm_pivot.rotation.x = lerp(left_arm_pivot.rotation.x, left_arm_target, delta * 10.0)
	right_arm_pivot.rotation.x = lerp(right_arm_pivot.rotation.x, right_arm_target, delta * 10.0)
	left_leg_pivot.rotation.x = lerp(left_leg_pivot.rotation.x, left_leg_target, delta * 10.0)
	right_leg_pivot.rotation.x = lerp(right_leg_pivot.rotation.x, right_leg_target, delta * 10.0)


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
		model_instance.name = "RobotKyleModel"
		if model_instance is Node3D:
			(model_instance as Node3D).position = Vector3(0.0, -0.9, 0.0)
			(model_instance as Node3D).scale = model_scale
		model_container.add_child(model_instance)
		return

	if resource is Mesh:
		var mesh_instance := MeshInstance3D.new()
		mesh_instance.name = "RobotKyleModel"
		mesh_instance.mesh = resource as Mesh
		mesh_instance.position = Vector3(0.0, -0.9, 0.0)
		mesh_instance.scale = model_scale
		model_container.add_child(mesh_instance)


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
