extends Node3D

@export var step_frequency: float = 1.9
@export var drive_force: float = 55.0
@export var lift_force: float = 18.0
@export var balance_torque: float = 35.0
@export var damping_force: float = 8.0

@onready var torso: RigidBody3D = $Agent/Torso
@onready var head: RigidBody3D = $Agent/Head
@onready var left_thigh: RigidBody3D = $Agent/LeftThigh
@onready var right_thigh: RigidBody3D = $Agent/RightThigh
@onready var left_shin: RigidBody3D = $Agent/LeftShin
@onready var right_shin: RigidBody3D = $Agent/RightShin
@onready var cam_pivot: Node3D = $CameraRig
@onready var info_label: Label = $CanvasLayer/InfoLabel

var _time: float = 0.0
var _camera_offset: Vector3 = Vector3(0.0, 4.8, 7.5)
var _respawn_cooldown: float = 0.0


func _ready() -> void:
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		body.can_sleep = false


func _physics_process(delta: float) -> void:
	if _respawn_cooldown > 0.0:
		_respawn_cooldown = maxf(_respawn_cooldown - delta, 0.0)

	if _needs_respawn():
		reset_pose()
		_respawn_cooldown = 0.35

	_time += delta
	_apply_balance()
	_apply_gait_forces()
	_apply_damping()
	_update_camera(delta)
	_update_info()


func _apply_balance() -> void:
	if _respawn_cooldown > 0.0:
		return
	var up := torso.global_basis.y.normalized()
	var axis := up.cross(Vector3.UP)
	torso.apply_torque(axis * balance_torque)


func _apply_gait_forces() -> void:
	if _respawn_cooldown > 0.0:
		return
	var cycle := _time * TAU * step_frequency
	var left_phase := sin(cycle)
	var right_phase := sin(cycle + PI)
	var forward := -torso.global_basis.z.normalized()
	var side := torso.global_basis.x.normalized()

	left_thigh.apply_central_force(forward * left_phase * drive_force + Vector3.UP * maxf(left_phase, 0.0) * lift_force)
	right_thigh.apply_central_force(forward * right_phase * drive_force + Vector3.UP * maxf(right_phase, 0.0) * lift_force)

	left_shin.apply_central_force(forward * left_phase * (drive_force * 0.55))
	right_shin.apply_central_force(forward * right_phase * (drive_force * 0.55))

	# Mild alternating yaw helps the body find a stable walking rhythm.
	torso.apply_torque(Vector3.UP * sin(cycle * 0.5) * 7.5)
	torso.apply_central_force(side * sin(cycle) * 2.5)


func _apply_damping() -> void:
	if _respawn_cooldown > 0.0:
		return
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		body.apply_central_force(-body.linear_velocity * damping_force)
		body.apply_torque(-body.angular_velocity * damping_force * 0.6)


func _update_camera(delta: float) -> void:
	var torso_target := torso.global_position
	if not _is_finite_vector3(torso_target):
		torso_target = Vector3(0.0, 1.35, 0.0)
	if torso_target.y < 0.45:
		torso_target.y = 0.45

	var desired := torso_target + _camera_offset
	desired.y = maxf(desired.y, 3.2)
	desired.z = maxf(desired.z, 4.5)

	cam_pivot.global_position = cam_pivot.global_position.lerp(desired, clampf(delta * 2.8, 0.0, 1.0))
	cam_pivot.look_at(torso_target + Vector3(0.0, 0.8, 0.0), Vector3.UP)

	if cam_pivot.global_position.y < 2.6:
		cam_pivot.global_position.y = 2.6


func _update_info() -> void:
	var speed: float = Vector2(torso.linear_velocity.x, torso.linear_velocity.z).length()
	if not is_finite(speed):
		speed = 0.0
	info_label.text = "Full-body physics locomotion\nVelocidade: %s m/s\nDica: ajuste step_frequency e drive_force no Inspector" % String.num(speed, 2)


func reset_pose() -> void:
	_time = 0.0
	torso.global_position = Vector3(0.0, 1.35, 0.0)
	head.global_position = Vector3(0.0, 2.0, 0.0)
	left_thigh.global_position = Vector3(-0.23, 0.92, 0.0)
	right_thigh.global_position = Vector3(0.23, 0.92, 0.0)
	left_shin.global_position = Vector3(-0.23, 0.37, 0.0)
	right_shin.global_position = Vector3(0.23, 0.37, 0.0)
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		body.linear_velocity = Vector3.ZERO
		body.angular_velocity = Vector3.ZERO


func _needs_respawn() -> bool:
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		if not _is_finite_vector3(body.global_position) or not _is_finite_vector3(body.linear_velocity):
			return true

	if torso.global_position.y < -3.0:
		return true

	if head.global_position.y < -3.0:
		return true

	return false


func _is_finite_vector3(value: Vector3) -> bool:
	return is_finite(value.x) and is_finite(value.y) and is_finite(value.z)
