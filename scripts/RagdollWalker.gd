extends Node3D

const OBS_SIZE := 8
const ACT_SIZE := 8
const POLICY_SIZE := OBS_SIZE * ACT_SIZE + ACT_SIZE

@export var move_force: float = 48.0
@export var jump_force: float = 14.5
@export var upright_torque: float = 42.0
@export var gait_torque: float = 16.0
@export var limb_torque: float = 10.0
@export var drag_force: float = 0.18
@export var spawn_height: float = 1.15

var walker_id: int = -1
var alive: bool = true
var finished: bool = false
var finish_time: float = 0.0
var survived_time: float = 0.0
var progress: float = 0.0
var last_distance: float = INF
var policy: PackedFloat32Array = PackedFloat32Array()

var _start_distance: float = 0.0
var _body_parts: Dictionary = {}
var _mesh_nodes: Array = []
var _rest_offsets: Dictionary = {}
var _shared_material: StandardMaterial3D


func _ready() -> void:
	_build_rig()
	set_tint(Color(0.82, 0.86, 0.9, 1.0))
	if policy.size() != POLICY_SIZE:
		policy = _random_policy()
	reset_walker(Vector3.ZERO, Vector3(0.0, spawn_height, 6.0), 0.0)


func set_policy(new_policy: PackedFloat32Array) -> void:
	if new_policy.size() == POLICY_SIZE:
		policy = PackedFloat32Array(new_policy)


func randomize_policy() -> void:
	policy = _random_policy()


func mutate_policy(sigma: float, mutate_probability: float) -> PackedFloat32Array:
	var child := PackedFloat32Array(policy)
	for i in range(child.size()):
		if randf() < mutate_probability:
			child[i] += randfn(0.0, sigma)
	return child


func set_tint(color: Color) -> void:
	if _shared_material == null:
		_shared_material = StandardMaterial3D.new()
		_shared_material.roughness = 0.48
		_shared_material.metallic = 0.18
	_shared_material.albedo_color = color


func reset_walker(spawn_position: Vector3, target_position: Vector3, initial_yaw: float = 0.0) -> void:
	global_position = spawn_position
	rotation = Vector3(0.0, initial_yaw, 0.0)
	alive = true
	finished = false
	finish_time = 0.0
	survived_time = 0.0
	progress = 0.0
	last_distance = spawn_position.distance_to(target_position)
	_start_distance = maxf(last_distance, 0.001)

	for part_name in _body_parts.keys():
		var part := _body_parts[part_name] as RigidBody3D
		if part == null:
			continue
		var offset: Vector3 = _rest_offsets.get(part_name, Vector3.ZERO)
		part.position = offset
		part.rotation = Vector3.ZERO
		part.linear_velocity = Vector3.ZERO
		part.angular_velocity = Vector3.ZERO
		part.sleeping = false


func step_walker(delta: float, target_position: Vector3, target_velocity: Vector3, episode_elapsed: float, episode_limit: float) -> void:
	if not alive or finished:
		_apply_drag()
		return

	survived_time += delta
	var observation := get_observation(target_position, target_velocity, episode_elapsed, episode_limit)
	var action := _policy_forward(observation)
	_apply_action(action, target_position)
	_apply_drag()
	_update_progress(target_position)

	if _should_fall_out() or episode_elapsed > episode_limit:
		alive = false


func get_observation(target_position: Vector3, target_velocity: Vector3, episode_elapsed: float, episode_limit: float) -> PackedFloat32Array:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	if hips == null:
		return PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	var offset := target_position - hips.global_position
	var local_offset := hips.global_transform.basis.inverse() * offset
	var local_velocity := hips.global_transform.basis.inverse() * hips.linear_velocity
	var upright := hips.global_basis.y.dot(Vector3.UP)
	var grounded := 1.0 if _is_grounded() else 0.0

	return PackedFloat32Array([
		local_offset.x,
		local_offset.z,
		offset.length(),
		local_velocity.x,
		local_velocity.z,
		upright,
		grounded,
		clampf(episode_elapsed / maxf(episode_limit, 0.001), 0.0, 1.0)
	])


func get_status(target_position: Vector3) -> Dictionary:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	if hips == null:
		return {}

	return {
		"height": hips.global_position.y,
		"speed": Vector2(hips.linear_velocity.x, hips.linear_velocity.z).length(),
		"vertical_speed": hips.linear_velocity.y,
		"distance": hips.global_position.distance_to(target_position),
		"alive": alive,
		"finished": finished,
		"progress": progress,
		"survived_time": survived_time
	}


func get_score(max_round_time: float) -> float:
	var score := progress * 3.5 + survived_time * 0.12
	if finished:
		score += 35.0 + maxf(0.0, (max_round_time - finish_time)) * 1.5
	if not alive:
		score -= 10.0
	return score


func is_in_play() -> bool:
	return alive and not finished


func _build_rig() -> void:
	if not _body_parts.is_empty():
		return

	_shared_material = StandardMaterial3D.new()
	_shared_material.roughness = 0.48
	_shared_material.metallic = 0.18

	_create_body_part("hips", _make_capsule_shape(0.26, 0.62), _make_capsule_mesh(0.26, 0.62), Vector3(0.0, 1.12, 0.0), 5.4)
	_create_body_part("spine", _make_capsule_shape(0.20, 0.58), _make_capsule_mesh(0.20, 0.58), Vector3(0.0, 1.60, 0.0), 3.2)
	_create_body_part("head", _make_sphere_shape(0.22), _make_sphere_mesh(0.22), Vector3(0.0, 2.12, 0.0), 1.2)

	_create_body_part("left_thigh", _make_capsule_shape(0.14, 0.55), _make_capsule_mesh(0.14, 0.55), Vector3(-0.17, 0.70, 0.0), 1.9)
	_create_body_part("left_shin", _make_capsule_shape(0.12, 0.50), _make_capsule_mesh(0.12, 0.50), Vector3(-0.17, 0.23, 0.0), 1.5)
	_create_body_part("left_foot", _make_box_shape(Vector3(0.16, 0.06, 0.30)), _make_box_mesh(Vector3(0.16, 0.06, 0.30)), Vector3(-0.17, 0.02, 0.07), 0.55)

	_create_body_part("right_thigh", _make_capsule_shape(0.14, 0.55), _make_capsule_mesh(0.14, 0.55), Vector3(0.17, 0.70, 0.0), 1.9)
	_create_body_part("right_shin", _make_capsule_shape(0.12, 0.50), _make_capsule_mesh(0.12, 0.50), Vector3(0.17, 0.23, 0.0), 1.5)
	_create_body_part("right_foot", _make_box_shape(Vector3(0.16, 0.06, 0.30)), _make_box_mesh(Vector3(0.16, 0.06, 0.30)), Vector3(0.17, 0.02, 0.07), 0.55)

	_create_body_part("left_arm", _make_capsule_shape(0.10, 0.42), _make_capsule_mesh(0.10, 0.42), Vector3(-0.32, 1.48, 0.0), 1.2)
	_create_body_part("left_forearm", _make_capsule_shape(0.09, 0.38), _make_capsule_mesh(0.09, 0.38), Vector3(-0.52, 1.22, 0.0), 0.9)
	_create_body_part("right_arm", _make_capsule_shape(0.10, 0.42), _make_capsule_mesh(0.10, 0.42), Vector3(0.32, 1.48, 0.0), 1.2)
	_create_body_part("right_forearm", _make_capsule_shape(0.09, 0.38), _make_capsule_mesh(0.09, 0.38), Vector3(0.52, 1.22, 0.0), 0.9)

	_create_joint("hips_spine_joint", Vector3(0.0, 1.35, 0.0), "hips", "spine")
	_create_joint("spine_head_joint", Vector3(0.0, 1.88, 0.0), "spine", "head")
	_create_joint("hips_left_thigh_joint", Vector3(-0.17, 0.93, 0.0), "hips", "left_thigh")
	_create_joint("left_knee_joint", Vector3(-0.17, 0.46, 0.0), "left_thigh", "left_shin")
	_create_joint("left_ankle_joint", Vector3(-0.17, 0.12, 0.11), "left_shin", "left_foot")
	_create_joint("hips_right_thigh_joint", Vector3(0.17, 0.93, 0.0), "hips", "right_thigh")
	_create_joint("right_knee_joint", Vector3(0.17, 0.46, 0.0), "right_thigh", "right_shin")
	_create_joint("right_ankle_joint", Vector3(0.17, 0.12, 0.11), "right_shin", "right_foot")
	_create_joint("spine_left_shoulder_joint", Vector3(-0.30, 1.53, 0.0), "spine", "left_arm")
	_create_joint("left_elbow_joint", Vector3(-0.44, 1.34, 0.0), "left_arm", "left_forearm")
	_create_joint("spine_right_shoulder_joint", Vector3(0.30, 1.53, 0.0), "spine", "right_arm")
	_create_joint("right_elbow_joint", Vector3(0.44, 1.34, 0.0), "right_arm", "right_forearm")

	_apply_tint_to_meshes()


func _create_body_part(name: String, shape: Shape3D, mesh: Mesh, local_position: Vector3, mass: float) -> void:
	var body := RigidBody3D.new()
	body.name = name.capitalize()
	body.mass = mass
	body.can_sleep = false
	body.contact_monitor = true
	body.max_contacts_reported = 8
	body.linear_damp = 0.22
	body.angular_damp = 0.30
	body.gravity_scale = 1.0
	body.position = local_position
	add_child(body)

	var collision := CollisionShape3D.new()
	collision.shape = shape
	body.add_child(collision)

	var mesh_instance := MeshInstance3D.new()
	mesh_instance.mesh = mesh
	mesh_instance.material_override = _shared_material
	body.add_child(mesh_instance)

	_body_parts[name] = body
	_rest_offsets[name] = local_position
	_mesh_nodes.append(mesh_instance)


func _create_joint(name: String, local_position: Vector3, parent_part: String, child_part: String) -> void:
	var joint := PinJoint3D.new()
	joint.name = name
	joint.position = local_position
	joint.node_a = NodePath("../%s" % parent_part.capitalize())
	joint.node_b = NodePath("../%s" % child_part.capitalize())
	add_child(joint)


func _apply_action(action: PackedFloat32Array, target_position: Vector3) -> void:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	var spine := _body_parts.get("spine", null) as RigidBody3D
	var head := _body_parts.get("head", null) as RigidBody3D
	var left_thigh := _body_parts.get("left_thigh", null) as RigidBody3D
	var left_shin := _body_parts.get("left_shin", null) as RigidBody3D
	var left_foot := _body_parts.get("left_foot", null) as RigidBody3D
	var right_thigh := _body_parts.get("right_thigh", null) as RigidBody3D
	var right_shin := _body_parts.get("right_shin", null) as RigidBody3D
	var right_foot := _body_parts.get("right_foot", null) as RigidBody3D
	var left_arm := _body_parts.get("left_arm", null) as RigidBody3D
	var left_forearm := _body_parts.get("left_forearm", null) as RigidBody3D
	var right_arm := _body_parts.get("right_arm", null) as RigidBody3D
	var right_forearm := _body_parts.get("right_forearm", null) as RigidBody3D

	if hips == null:
		return

	var move_x := clampf(action[0], -1.0, 1.0)
	var move_z := clampf(action[1], -1.0, 1.0)
	var jump := action[2]
	var torso_turn := clampf(action[3], -1.0, 1.0)
	var arm_drive := clampf(action[4], -1.0, 1.0)
	var leg_drive := clampf(action[5], -1.0, 1.0)
	var balance_drive := clampf(action[6], -1.0, 1.0)
	var strength := clampf(action[7] * 0.5 + 0.5, 0.0, 1.0)

	var to_target := target_position - hips.global_position
	to_target.y = 0.0
	if to_target.length_squared() > 0.0001:
		to_target = to_target.normalized()

	var forward := -hips.global_basis.z.normalized()
	var lateral := hips.global_basis.x.normalized()
	var desired_drive := forward * move_z + lateral * move_x + to_target * 0.85
	if desired_drive.length_squared() > 0.0001:
		desired_drive = desired_drive.normalized()

	var drive_scale := move_force * (0.45 + strength * 0.8)
	hips.apply_central_force(desired_drive * drive_scale)
	hips.apply_central_force(Vector3.UP * maxf(0.0, jump) * jump_force * strength)
	hips.apply_central_force(-hips.linear_velocity * drag_force)

	var upright_axis := hips.global_basis.y.cross(Vector3.UP)
	hips.apply_torque(upright_axis * upright_torque * (0.45 + strength * 0.55))
	hips.apply_torque(Vector3.UP * torso_turn * upright_torque * 0.35)
	hips.apply_torque(Vector3.RIGHT * balance_drive * upright_torque * 0.12)

	if spine != null:
		spine.apply_torque(Vector3.RIGHT * torso_turn * gait_torque * 0.65)
		spine.apply_torque(Vector3.FORWARD * balance_drive * gait_torque * 0.45)

	if head != null:
		head.apply_torque(Vector3.RIGHT * arm_drive * gait_torque * 0.12)
		head.apply_torque(Vector3.UP * torso_turn * gait_torque * 0.08)

	var gait := sin(survived_time * 7.0 + torso_turn * 1.4)
	var inverse_gait := -gait

	if left_arm != null:
		left_arm.apply_torque(Vector3.RIGHT * (gait + arm_drive) * limb_torque)
	if left_forearm != null:
		left_forearm.apply_torque(Vector3.RIGHT * (gait + arm_drive * 0.6) * limb_torque * 0.65)
	if right_arm != null:
		right_arm.apply_torque(Vector3.RIGHT * (inverse_gait - arm_drive) * limb_torque)
	if right_forearm != null:
		right_forearm.apply_torque(Vector3.RIGHT * (inverse_gait - arm_drive * 0.6) * limb_torque * 0.65)

	if left_thigh != null:
		left_thigh.apply_torque(Vector3.RIGHT * (inverse_gait - leg_drive) * limb_torque * 1.1)
	if left_shin != null:
		left_shin.apply_torque(Vector3.RIGHT * (inverse_gait - leg_drive * 0.5) * limb_torque * 0.85)
	if left_foot != null:
		left_foot.apply_torque(Vector3.RIGHT * (inverse_gait - leg_drive * 0.2) * limb_torque * 0.45)

	if right_thigh != null:
		right_thigh.apply_torque(Vector3.RIGHT * (gait + leg_drive) * limb_torque * 1.1)
	if right_shin != null:
		right_shin.apply_torque(Vector3.RIGHT * (gait + leg_drive * 0.5) * limb_torque * 0.85)
	if right_foot != null:
		right_foot.apply_torque(Vector3.RIGHT * (gait + leg_drive * 0.2) * limb_torque * 0.45)


func _apply_drag() -> void:
	for part_variant in _body_parts.values():
		var part := part_variant as RigidBody3D
		if part == null:
			continue
		part.apply_central_force(-part.linear_velocity * drag_force)
		part.apply_torque(-part.angular_velocity * drag_force * 0.65)


func _update_progress(target_position: Vector3) -> void:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	if hips == null:
		return

	var distance := hips.global_position.distance_to(target_position)
	last_distance = distance
	progress = maxf(progress, _start_distance - distance)
	if not is_finite(progress):
		progress = 0.0


func _should_fall_out() -> bool:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	var head := _body_parts.get("head", null) as RigidBody3D
	if hips == null or head == null:
		return true

	if hips.global_position.y < -4.0:
		return true

	if head.global_position.y < -4.0:
		return true

	if not _is_finite_vector3(hips.global_position):
		return true

	if not _is_finite_vector3(hips.linear_velocity):
		return true

	return false


func _is_grounded() -> bool:
	var hips := _body_parts.get("hips", null) as RigidBody3D
	if hips == null:
		return false
	return hips.global_position.y <= spawn_height + 0.12


func _policy_forward(obs: PackedFloat32Array) -> PackedFloat32Array:
	if policy.size() != POLICY_SIZE:
		policy = _random_policy()

	var out := PackedFloat32Array()
	out.resize(ACT_SIZE)
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
		p[i] = randf_range(-0.55, 0.55)
	return p


func _apply_tint_to_meshes() -> void:
	for mesh_variant in _mesh_nodes:
		var mesh := mesh_variant as MeshInstance3D
		if mesh != null:
			mesh.material_override = _shared_material


func _make_capsule_shape(radius: float, height: float) -> CapsuleShape3D:
	var shape := CapsuleShape3D.new()
	shape.radius = radius
	shape.height = height
	return shape


func _make_capsule_mesh(radius: float, height: float) -> CapsuleMesh:
	var mesh := CapsuleMesh.new()
	mesh.radius = radius
	mesh.height = height
	return mesh


func _make_sphere_shape(radius: float) -> SphereShape3D:
	var shape := SphereShape3D.new()
	shape.radius = radius
	return shape


func _make_sphere_mesh(radius: float) -> SphereMesh:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	return mesh


func _make_box_shape(size: Vector3) -> BoxShape3D:
	var shape := BoxShape3D.new()
	shape.size = size
	return shape


func _make_box_mesh(size: Vector3) -> BoxMesh:
	var mesh := BoxMesh.new()
	mesh.size = size
	return mesh


func _is_finite_vector3(value: Vector3) -> bool:
	return is_finite(value.x) and is_finite(value.y) and is_finite(value.z)