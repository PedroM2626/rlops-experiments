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


func _ready() -> void:
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		body.can_sleep = false


func _physics_process(delta: float) -> void:
	_time += delta
	_apply_balance()
	_apply_gait_forces()
	_apply_damping()
	_update_camera(delta)
	_update_info()


func _apply_balance() -> void:
	var up := torso.global_basis.y.normalized()
	var axis := up.cross(Vector3.UP)
	torso.apply_torque(axis * balance_torque)


func _apply_gait_forces() -> void:
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
	for body in [torso, head, left_thigh, right_thigh, left_shin, right_shin]:
		body.apply_central_force(-body.linear_velocity * damping_force)
		body.apply_torque(-body.angular_velocity * damping_force * 0.6)


func _update_camera(delta: float) -> void:
	var desired := torso.global_position + Vector3(0.0, 4.8, 7.5)
	cam_pivot.global_position = cam_pivot.global_position.lerp(desired, clampf(delta * 2.8, 0.0, 1.0))
	cam_pivot.look_at(torso.global_position + Vector3(0.0, 0.7, 0.0), Vector3.UP)


func _update_info() -> void:
	var speed := Vector2(torso.linear_velocity.x, torso.linear_velocity.z).length()
	info_label.text = "Full-body physics locomotion\nVelocidade: %s m/s\nDica: ajuste step_frequency e drive_force no Inspector" % String.num(speed, 2)
