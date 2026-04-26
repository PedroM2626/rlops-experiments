extends Node3D

const AGENT_COUNT: int = 6
const ROUND_TIME_LIMIT: float = 40.0
const POLICY_SIZE: int = 21
const SAVE_DIR := "user://parkour_competition"
const POLICIES_PATH := SAVE_DIR + "/policies.json"
const RESULTS_PATH := SAVE_DIR + "/results.json"
const TRAINING_LOG_PATH := SAVE_DIR + "/training_log.jsonl"
const RUNNER_SCRIPT: GDScript = preload("res://scripts/CompetitiveRunner.gd")

@onready var agents_root: Node3D = $Agents
@onready var camera_rig: Node3D = $CameraRig
@onready var ranking_label: RichTextLabel = $CanvasLayer/HUD/Panel/VBox/Ranking
@onready var generation_label: Label = $CanvasLayer/HUD/Panel/VBox/Generation
@onready var timer_label: Label = $CanvasLayer/HUD/Panel/VBox/Timer
@onready var train_button: Button = $CanvasLayer/HUD/Panel/VBox/Buttons/TrainButton
@onready var reset_button: Button = $CanvasLayer/HUD/Panel/VBox/Buttons/ResetButton
@onready var save_label: Label = $CanvasLayer/HUD/Panel/VBox/SaveInfo

var generation: int = 0
var elapsed: float = 0.0
var finish_z: float = 42.0
var lane_x_positions: Array[float] = [-6.0, -3.6, -1.2, 1.2, 3.6, 6.0]
var obstacle_z_positions: Array[float] = [8.0, 14.0, 20.0, 27.0, 35.0]
var obstacle_heights: Array[float] = [1.0, 1.2, 1.1, 1.35, 1.25]

var runners: Array = []
var policies: Array[PackedFloat32Array] = []
var last_scores: Array[float] = []


func _ready() -> void:
	randomize()
	train_button.pressed.connect(_on_train_pressed)
	reset_button.pressed.connect(_on_reset_pressed)

	_create_course()
	_create_runners()
	_load_saved_state()
	_start_round()


func _physics_process(delta: float) -> void:
	elapsed += delta
	var all_stopped := true

	for i in range(runners.size()):
		var runner: Variant = runners[i]
		var next_info := _next_obstacle_info(runner.global_position.z)
		runner.step_runner(
			delta,
			next_info["z"],
			next_info["h"],
			lane_x_positions[i],
			finish_z
		)
		if runner.alive and not runner.finished:
			all_stopped = false

	if elapsed >= ROUND_TIME_LIMIT:
		all_stopped = true

	if all_stopped:
		_finalize_round()

	_update_camera(delta)
	_update_hud()


func _create_course() -> void:
	var ground_body := StaticBody3D.new()
	ground_body.name = "Ground"
	add_child(ground_body)

	var collision := CollisionShape3D.new()
	var ground_shape := BoxShape3D.new()
	ground_shape.size = Vector3(40.0, 1.0, 120.0)
	collision.shape = ground_shape
	collision.position = Vector3(0.0, -0.5, 25.0)
	ground_body.add_child(collision)

	var ground_mesh := MeshInstance3D.new()
	var box_mesh := BoxMesh.new()
	box_mesh.size = Vector3(40.0, 1.0, 120.0)
	ground_mesh.mesh = box_mesh
	ground_mesh.position = Vector3(0.0, -0.5, 25.0)
	var ground_mat := StandardMaterial3D.new()
	ground_mat.albedo_color = Color(0.2, 0.35, 0.2)
	ground_mat.roughness = 0.85
	ground_mesh.material_override = ground_mat
	ground_body.add_child(ground_mesh)

	for idx in range(obstacle_z_positions.size()):
		var z := obstacle_z_positions[idx]
		var h := obstacle_heights[idx]
		for lane_x in lane_x_positions:
			var obs := StaticBody3D.new()
			add_child(obs)
			var obs_shape := CollisionShape3D.new()
			var obs_box := BoxShape3D.new()
			obs_box.size = Vector3(1.4, h, 1.2)
			obs_shape.shape = obs_box
			obs_shape.position = Vector3(lane_x, h * 0.5, z)
			obs.add_child(obs_shape)
			var obs_mesh := MeshInstance3D.new()
			var obs_box_mesh := BoxMesh.new()
			obs_box_mesh.size = Vector3(1.4, h, 1.2)
			obs_mesh.mesh = obs_box_mesh
			obs_mesh.position = Vector3(lane_x, h * 0.5, z)
			var obs_mat := StandardMaterial3D.new()
			obs_mat.albedo_color = Color(0.75, 0.35 + float(idx) * 0.08, 0.2)
			obs_mesh.material_override = obs_mat
			obs.add_child(obs_mesh)


func _create_runners() -> void:
	runners.clear()
	for i in range(AGENT_COUNT):
		var runner: CharacterBody3D = _build_runner_node(i)
		runner.runner_id = i
		agents_root.add_child(runner)
		runners.append(runner)
		policies.append(_random_policy())
		last_scores.append(0.0)


func _build_runner_node(index: int) -> CharacterBody3D:
	var runner: CharacterBody3D = RUNNER_SCRIPT.new()
	runner.name = "Runner_%d" % index

	var col := CollisionShape3D.new()
	var cap := CapsuleShape3D.new()
	cap.radius = 0.27
	cap.height = 1.0
	col.shape = cap
	col.position = Vector3(0.0, 0.95, 0.0)
	runner.add_child(col)

	var body_root := Node3D.new()
	body_root.name = "BodyRoot"
	runner.add_child(body_root)

	var torso := MeshInstance3D.new()
	torso.name = "Torso"
	var torso_mesh := CapsuleMesh.new()
	torso_mesh.radius = 0.24
	torso_mesh.height = 0.85
	torso.mesh = torso_mesh
	torso.position = Vector3(0.0, 1.1, 0.0)
	body_root.add_child(torso)

	var head_pivot := Node3D.new()
	head_pivot.name = "HeadPivot"
	head_pivot.position = Vector3(0.0, 1.65, 0.0)
	body_root.add_child(head_pivot)

	var head := MeshInstance3D.new()
	head.name = "Head"
	var head_mesh := SphereMesh.new()
	head_mesh.radius = 0.17
	head.mesh = head_mesh
	head_pivot.add_child(head)

	var left_arm_pivot := Node3D.new()
	left_arm_pivot.name = "LeftArmPivot"
	left_arm_pivot.position = Vector3(-0.33, 1.34, 0.0)
	body_root.add_child(left_arm_pivot)

	var left_arm := MeshInstance3D.new()
	left_arm.name = "LeftArm"
	var arm_mesh := CapsuleMesh.new()
	arm_mesh.radius = 0.08
	arm_mesh.height = 0.42
	left_arm.mesh = arm_mesh
	left_arm.position = Vector3(0.0, -0.24, 0.0)
	left_arm_pivot.add_child(left_arm)

	var right_arm_pivot := Node3D.new()
	right_arm_pivot.name = "RightArmPivot"
	right_arm_pivot.position = Vector3(0.33, 1.34, 0.0)
	body_root.add_child(right_arm_pivot)

	var right_arm := MeshInstance3D.new()
	right_arm.name = "RightArm"
	right_arm.mesh = arm_mesh
	right_arm.position = Vector3(0.0, -0.24, 0.0)
	right_arm_pivot.add_child(right_arm)

	var left_leg_pivot := Node3D.new()
	left_leg_pivot.name = "LeftLegPivot"
	left_leg_pivot.position = Vector3(-0.14, 0.62, 0.0)
	body_root.add_child(left_leg_pivot)

	var left_leg := MeshInstance3D.new()
	left_leg.name = "LeftLeg"
	var leg_mesh := CapsuleMesh.new()
	leg_mesh.radius = 0.09
	leg_mesh.height = 0.48
	left_leg.mesh = leg_mesh
	left_leg.position = Vector3(0.0, -0.28, 0.0)
	left_leg_pivot.add_child(left_leg)

	var right_leg_pivot := Node3D.new()
	right_leg_pivot.name = "RightLegPivot"
	right_leg_pivot.position = Vector3(0.14, 0.62, 0.0)
	body_root.add_child(right_leg_pivot)

	var right_leg := MeshInstance3D.new()
	right_leg.name = "RightLeg"
	right_leg.mesh = leg_mesh
	right_leg.position = Vector3(0.0, -0.28, 0.0)
	right_leg_pivot.add_child(right_leg)

	var hue := float(index) / float(max(AGENT_COUNT, 1))
	runner.set_tint(Color.from_hsv(hue, 0.7, 0.95, 1.0))
	return runner


func _next_obstacle_info(agent_z: float) -> Dictionary:
	for i in range(obstacle_z_positions.size()):
		if obstacle_z_positions[i] > agent_z:
			return {"z": obstacle_z_positions[i], "h": obstacle_heights[i]}
	return {"z": finish_z + 5.0, "h": 0.0}


func _start_round() -> void:
	elapsed = 0.0
	for i in range(runners.size()):
		var spawn := Vector3(lane_x_positions[i], 0.2, 0.0)
		runners[i].reset_runner(spawn, policies[i])
		last_scores[i] = 0.0


func _finalize_round() -> void:
	for i in range(runners.size()):
		last_scores[i] = runners[i].get_score(ROUND_TIME_LIMIT)

	_save_results_snapshot()
	_save_training_log()
	_evolve_generation()
	_start_round()


func _evolve_generation() -> void:
	generation += 1
	var order := _score_order_desc(last_scores)
	var elite_a := policies[order[0]]
	var elite_b := policies[order[1]]

	var next_policies: Array[PackedFloat32Array] = []
	next_policies.resize(AGENT_COUNT)
	next_policies[0] = PackedFloat32Array(elite_a)
	next_policies[1] = PackedFloat32Array(elite_b)

	for i in range(2, AGENT_COUNT):
		var parent := elite_a if i % 2 == 0 else elite_b
		next_policies[i] = _mutate_policy(parent, 0.08, 0.22)

	policies = next_policies
	_save_policies()


func _score_order_desc(scores: Array[float]) -> Array[int]:
	var indices: Array[int] = []
	for i in range(scores.size()):
		indices.append(i)
	indices.sort_custom(func(a: int, b: int) -> bool: return scores[a] > scores[b])
	return indices


func _mutate_policy(parent: PackedFloat32Array, sigma: float, mutate_prob: float) -> PackedFloat32Array:
	var child := PackedFloat32Array(parent)
	for i in range(child.size()):
		if randf() < mutate_prob:
			child[i] += randfn(0.0, sigma)
	return child


func _random_policy() -> PackedFloat32Array:
	var p := PackedFloat32Array()
	p.resize(POLICY_SIZE)
	for i in range(p.size()):
		p[i] = randf_range(-0.6, 0.6)
	return p


func _update_camera(delta: float) -> void:
	var avg_pos := Vector3.ZERO
	for runner in runners:
		avg_pos += runner.global_position
	avg_pos /= float(max(runners.size(), 1))
	var desired := avg_pos + Vector3(0.0, 12.0, -16.0)
	camera_rig.global_position = camera_rig.global_position.lerp(desired, clampf(delta * 1.8, 0.0, 1.0))
	camera_rig.look_at(avg_pos + Vector3(0.0, 1.0, 6.0), Vector3.UP)


func _update_hud() -> void:
	generation_label.text = "Geracao: %d" % generation
	timer_label.text = "Tempo da rodada: %s / %s" % [String.num(elapsed, 2), String.num(ROUND_TIME_LIMIT, 2)]

	var lines: Array[String] = []
	var order := _score_order_desc(last_scores)
	for rank in range(order.size()):
		var idx := order[rank]
		var r: Variant = runners[idx]
		var status := "running"
		if r.finished:
			status = "finished"
		elif not r.alive:
			status = "down"
		lines.append("%d) IA-%d | score=%s | z=%s | %s" % [
			rank + 1,
			idx,
			String.num(last_scores[idx], 2),
			String.num(r.progress, 2),
			status
		])

	ranking_label.text = "\n".join(lines)


func _on_train_pressed() -> void:
	_finalize_round()


func _on_reset_pressed() -> void:
	generation = 0
	for i in range(policies.size()):
		policies[i] = _random_policy()
	_save_policies()
	_start_round()


func _load_saved_state() -> void:
	_ensure_save_dir()
	if FileAccess.file_exists(ProjectSettings.globalize_path(POLICIES_PATH)):
		var parser := JSON.new()
		var content := FileAccess.get_file_as_string(POLICIES_PATH)
		if parser.parse(content) == OK and typeof(parser.data) == TYPE_DICTIONARY:
			var data: Dictionary = parser.data
			generation = int(data.get("generation", 0))
			var saved: Variant = data.get("policies", [])
			if typeof(saved) == TYPE_ARRAY:
				var arr: Array = saved
				for i in range(min(arr.size(), policies.size())):
					var raw: Variant = arr[i]
					if typeof(raw) == TYPE_ARRAY:
						var p := PackedFloat32Array()
						for v in raw:
							p.append(float(v))
						if p.size() == POLICY_SIZE:
							policies[i] = p

	_save_policies()
	_save_results_snapshot()


func _save_policies() -> void:
	_ensure_save_dir()
	var serialized: Array = []
	for p in policies:
		var arr: Array = []
		for value in p:
			arr.append(value)
		serialized.append(arr)

	var payload := {
		"generation": generation,
		"policies": serialized,
		"updated_at": Time.get_datetime_string_from_system()
	}
	var file := FileAccess.open(POLICIES_PATH, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(payload, "  "))
		save_label.text = "Saves: %s" % ProjectSettings.globalize_path(SAVE_DIR)


func _save_results_snapshot() -> void:
	_ensure_save_dir()
	var entries: Array = []
	for i in range(runners.size()):
		entries.append({
			"runner": i,
			"score": last_scores[i],
			"progress": runners[i].progress,
			"finished": runners[i].finished,
			"finish_time": runners[i].finish_time
		})

	var payload := {
		"generation": generation,
		"elapsed": elapsed,
		"entries": entries,
		"updated_at": Time.get_datetime_string_from_system()
	}
	var file := FileAccess.open(RESULTS_PATH, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(payload, "  "))


func _save_training_log() -> void:
	_ensure_save_dir()
	var payload := {
		"generation": generation,
		"scores": last_scores,
		"updated_at": Time.get_datetime_string_from_system()
	}
	var file := FileAccess.open(TRAINING_LOG_PATH, FileAccess.WRITE_READ)
	if file != null:
		file.seek_end()
		file.store_line(JSON.stringify(payload))


func _ensure_save_dir() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(SAVE_DIR))
