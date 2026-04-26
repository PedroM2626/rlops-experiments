extends CharacterBody3D
class_name CompetitiveRunner

const OBS_SIZE: int = 6
const ACT_SIZE: int = 3
const POLICY_SIZE: int = OBS_SIZE * ACT_SIZE + ACT_SIZE

@export var move_speed: float = 7.2
@export var lateral_speed: float = 3.0
@export var jump_velocity: float = 6.1
@export var gravity: float = 18.0

@onready var body_root: Node3D = $BodyRoot
@onready var torso: MeshInstance3D = $BodyRoot/Torso
@onready var head_pivot: Node3D = $BodyRoot/HeadPivot
@onready var left_arm_pivot: Node3D = $BodyRoot/LeftArmPivot
@onready var right_arm_pivot: Node3D = $BodyRoot/RightArmPivot
@onready var left_leg_pivot: Node3D = $BodyRoot/LeftLegPivot
@onready var right_leg_pivot: Node3D = $BodyRoot/RightLegPivot

var runner_id: int = -1
var policy: PackedFloat32Array = PackedFloat32Array()
var alive: bool = true
var finished: bool = false
var finish_time: float = 0.0
var progress: float = 0.0
var _elapsed: float = 0.0
var _pending_tint: Color = Color(1.0, 1.0, 1.0, 1.0)
var _has_pending_tint: bool = false


func _ready() -> void:
	if policy.size() != POLICY_SIZE:
		policy = _random_policy()
	if _has_pending_tint:
		_apply_tint(_pending_tint)
		_has_pending_tint = false


func reset_runner(spawn_position: Vector3, new_policy: PackedFloat32Array) -> void:
	global_position = spawn_position
	velocity = Vector3.ZERO
	alive = true
	finished = false
	finish_time = 0.0
	progress = 0.0
	_elapsed = 0.0
	policy = PackedFloat32Array(new_policy)
	body_root.rotation = Vector3.ZERO
	head_pivot.rotation = Vector3.ZERO
	left_arm_pivot.rotation = Vector3.ZERO
	right_arm_pivot.rotation = Vector3.ZERO
	left_leg_pivot.rotation = Vector3.ZERO
	right_leg_pivot.rotation = Vector3.ZERO


func step_runner(delta: float, next_obstacle_z: float, obstacle_height: float, lane_x: float, finish_z: float) -> void:
	if not alive or finished:
		return

	_elapsed += delta
	progress = maxf(progress, global_position.z)

	var obs := PackedFloat32Array([
		next_obstacle_z - global_position.z,
		obstacle_height,
		velocity.z,
		1.0 if is_on_floor() else 0.0,
		lane_x - global_position.x,
		clampf(_elapsed / 40.0, 0.0, 1.0)
	])
	var action := _policy_forward(obs)

	var drive := clampf(action[0] * 0.5 + 0.5, 0.0, 1.0)
	var jump_drive := action[1]
	var lateral := clampf(action[2], -1.0, 1.0)

	var desired := Vector3(lateral * lateral_speed, 0.0, drive * move_speed)
	velocity.x = move_toward(velocity.x, desired.x, delta * 10.0)
	velocity.z = move_toward(velocity.z, desired.z, delta * 10.0)

	if not is_on_floor():
		velocity.y = maxf(velocity.y - gravity * delta, -32.0)
	elif jump_drive > 0.25:
		velocity.y = jump_velocity

	if Vector2(velocity.x, velocity.z).length() > 0.2:
		var target_yaw := atan2(velocity.x, velocity.z)
		body_root.rotation.y = lerp_angle(body_root.rotation.y, target_yaw, delta * 7.0)

	_animate_limbs(delta, drive)
	move_and_slide()
	progress = maxf(progress, global_position.z)

	if global_position.y < -3.0:
		alive = false

	if global_position.z >= finish_z:
		finished = true
		finish_time = _elapsed


func get_score(max_round_time: float) -> float:
	var base := progress
	if finished:
		base += 100.0 + maxf(0.0, (max_round_time - finish_time) * 3.0)
	if not alive:
		base -= 12.0
	return base


func set_tint(color: Color) -> void:
	_pending_tint = color
	if is_inside_tree() and torso != null:
		_apply_tint(color)
		_has_pending_tint = false
		return
	_has_pending_tint = true


func _apply_tint(color: Color) -> void:
	var target_torso: MeshInstance3D = torso
	if target_torso == null:
		var found: Node = get_node_or_null("BodyRoot/Torso")
		if found == null:
			return
		target_torso = found as MeshInstance3D
		if target_torso == null:
			return
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	target_torso.material_override = mat


func _animate_limbs(delta: float, drive: float) -> void:
	var gait := sin(_elapsed * 8.0) * drive
	left_arm_pivot.rotation.x = lerpf(left_arm_pivot.rotation.x, gait * 0.85, delta * 10.0)
	right_arm_pivot.rotation.x = lerpf(right_arm_pivot.rotation.x, -gait * 0.85, delta * 10.0)
	left_leg_pivot.rotation.x = lerpf(left_leg_pivot.rotation.x, -gait * 1.0, delta * 10.0)
	right_leg_pivot.rotation.x = lerpf(right_leg_pivot.rotation.x, gait * 1.0, delta * 10.0)
	head_pivot.rotation.x = lerpf(head_pivot.rotation.x, clampf(velocity.y * 0.03, -0.2, 0.2), delta * 7.0)


func _policy_forward(obs: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array([0.0, 0.0, 0.0])
	for a in range(ACT_SIZE):
		var sum := policy[OBS_SIZE * ACT_SIZE + a]
		for o in range(OBS_SIZE):
			sum += policy[a * OBS_SIZE + o] * obs[o]
		out[a] = tanh(sum)
	return out


func _random_policy() -> PackedFloat32Array:
	var p := PackedFloat32Array()
	p.resize(POLICY_SIZE)
	for i in range(POLICY_SIZE):
		p[i] = randf_range(-0.6, 0.6)
	return p
