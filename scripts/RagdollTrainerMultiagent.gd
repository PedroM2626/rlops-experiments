extends Node3D
class_name RagdollTrainerMultiagent

const RAGDOLL_WALKER_PATH := "res://scripts/RagdollWalker.gd"
const WALKER_POLICY_SIZE := 72

@export var agent_count: int = 4
@export var round_time_limit: float = 45.0
@export var capture_distance: float = 1.55
@export var arena_radius: float = 18.0
@export var spawn_radius: float = 10.5
@export var mutation_sigma: float = 0.08
@export var mutation_probability: float = 0.20
@export var winner_bonus: float = 2.0
@export var participant_bonus: float = 0.1

@onready var arena_root: Node3D = $ArenaRoot
@onready var camera_rig: Node3D = $CameraRig
@onready var info_label: Label = $CanvasLayer/HUD/Panel/VBox/InfoLabel
@onready var ranking_label: RichTextLabel = $CanvasLayer/HUD/Panel/VBox/RankingLabel
@onready var reset_button: Button = $CanvasLayer/HUD/Panel/VBox/ButtonsRow/ResetButton
@onready var shuffle_button: Button = $CanvasLayer/HUD/Panel/VBox/ButtonsRow/ShuffleButton
@onready var menu_button: Button = $CanvasLayer/HUD/Panel/VBox/ButtonsRow/MenuButton

var walkers: Array = []
var policies: Array = []
var scores: Array = []
var generation: int = 0
var round_elapsed: float = 0.0
var round_reason: String = "Aguardando"
var round_active: bool = false
var target_node: Node3D = null
var target_position: Vector3 = Vector3.ZERO
var winner_index: int = -1
var best_score: float = 0.0


func _ready() -> void:
	randomize()
	reset_button.pressed.connect(_on_reset_pressed)
	shuffle_button.pressed.connect(_on_shuffle_pressed)
	menu_button.pressed.connect(_on_menu_pressed)

	_build_arena()
	_create_target()
	_create_walkers()
	call_deferred("_start_round")


func _physics_process(delta: float) -> void:
	if target_node == null or walkers.is_empty():
		return

	if round_active:
		round_elapsed += delta
		for i in range(walkers.size()):
			var walker = walkers[i]
			if walker == null:
				continue
			walker.step_walker(delta, target_position, Vector3.ZERO, round_elapsed, round_time_limit)
			var status: Dictionary = walker.get_status(target_position)
			if bool(status.get("alive", false)) and not bool(status.get("finished", false)):
				if float(status.get("distance", INF)) <= capture_distance:
					winner_index = i
					_finish_round("capturado")
					break

		if round_active and round_elapsed >= round_time_limit:
			_finish_round("tempo limite")

		if round_active and _all_walkers_inactive():
			_finish_round("todos cairam")

	_update_camera(delta)
	_update_hud()


func _build_arena() -> void:
	var environment := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	env.background_color = Color(0.62, 0.78, 0.89, 1.0)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_color = Color(0.72, 0.75, 0.80, 1.0)
	env.ambient_light_energy = 1.0
	environment.environment = env
	add_child(environment)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-45.0, -35.0, 0.0)
	sun.light_energy = 2.5
	sun.shadow_enabled = true
	add_child(sun)

	var ground := StaticBody3D.new()
	ground.name = "Ground"
	arena_root.add_child(ground)

	var ground_shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = Vector3(arena_radius * 2.4, 1.0, arena_radius * 2.4)
	ground_shape.shape = box_shape
	ground_shape.position = Vector3(0.0, -0.5, 0.0)
	ground.add_child(ground_shape)

	var ground_mesh := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = Vector3(arena_radius * 2.4, 1.0, arena_radius * 2.4)
	ground_mesh.mesh = mesh
	ground_mesh.position = Vector3(0.0, -0.5, 0.0)
	var ground_material := StandardMaterial3D.new()
	ground_material.albedo_color = Color(0.22, 0.34, 0.38, 1.0)
	ground_material.roughness = 0.88
	ground_mesh.material_override = ground_material
	ground.add_child(ground_mesh)

	var obstacle_specs := [
		{"size": Vector3(3.0, 1.4, 6.0), "pos": Vector3(-6.0, 0.7, 4.5), "color": Color(0.80, 0.42, 0.24, 1.0)},
		{"size": Vector3(4.0, 2.0, 3.0), "pos": Vector3(5.5, 1.0, -3.0), "color": Color(0.76, 0.52, 0.28, 1.0)},
		{"size": Vector3(2.0, 1.0, 8.0), "pos": Vector3(-1.0, 0.5, -8.5), "color": Color(0.68, 0.48, 0.26, 1.0)}
	]
	for spec_variant in obstacle_specs:
		var spec := spec_variant as Dictionary
		var obstacle := StaticBody3D.new()
		arena_root.add_child(obstacle)
		var obstacle_shape := CollisionShape3D.new()
		var obstacle_box := BoxShape3D.new()
		obstacle_box.size = spec["size"]
		obstacle_shape.shape = obstacle_box
		obstacle_shape.position = spec["pos"] + Vector3(0.0, spec["size"].y * 0.5, 0.0)
		obstacle.add_child(obstacle_shape)
		var obstacle_mesh := MeshInstance3D.new()
		var obstacle_mesh_box := BoxMesh.new()
		obstacle_mesh_box.size = spec["size"]
		obstacle_mesh.mesh = obstacle_mesh_box
		obstacle_mesh.position = obstacle_shape.position
		var obstacle_material := StandardMaterial3D.new()
		obstacle_material.albedo_color = spec["color"]
		obstacle_material.roughness = 0.82
		obstacle_mesh.material_override = obstacle_material
		obstacle.add_child(obstacle_mesh)


func _create_target() -> void:
	target_node = Node3D.new()
	target_node.name = "Target"
	arena_root.add_child(target_node)

	var target_mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.45
	target_mesh.mesh = sphere
	var target_material := StandardMaterial3D.new()
	target_material.albedo_color = Color(0.96, 0.91, 0.42, 1.0)
	target_material.emission_enabled = true
	target_material.emission = Color(0.7, 0.62, 0.18, 1.0)
	target_material.emission_energy_multiplier = 1.6
	target_mesh.material_override = target_material
	target_node.add_child(target_mesh)

	var target_area := Area3D.new()
	target_node.add_child(target_area)
	var target_collision := CollisionShape3D.new()
	var target_shape := SphereShape3D.new()
	target_shape.radius = capture_distance
	target_collision.shape = target_shape
	target_area.add_child(target_collision)

	_move_target_to_random_position()


func _create_walkers() -> void:
	walkers.clear()
	policies.clear()
	scores.clear()
	var walker_script := load(RAGDOLL_WALKER_PATH)
	if walker_script == null:
		push_error("Nao foi possivel carregar o script do walker: %s" % RAGDOLL_WALKER_PATH)
		return
	for i in range(agent_count):
		var walker = walker_script.new()
		walker.name = "Walker_%d" % i
		walker.walker_id = i
		arena_root.add_child(walker)
		if walker.policy.size() != WALKER_POLICY_SIZE:
			walker.randomize_policy()
		walkers.append(walker)
		policies.append(PackedFloat32Array(walker.policy))
		scores.append(0.0)
		var tint := Color.from_hsv(float(i) / float(max(agent_count, 1)), 0.72, 0.95, 1.0)
		walker.set_tint(tint)


func _start_round() -> void:
	_move_target_to_random_position()
	round_elapsed = 0.0
	round_reason = "Rodando"
	winner_index = -1
	round_active = true

	for i in range(walkers.size()):
		var walker = walkers[i]
		if walker == null:
			continue
		var spawn_angle := TAU * float(i) / float(max(walkers.size(), 1)) + randf_range(-0.25, 0.25)
		var spawn := Vector3(cos(spawn_angle) * spawn_radius, 1.2, sin(spawn_angle) * spawn_radius)
		walker.set_policy(policies[i])
		walker.reset_walker(spawn, target_position, spawn_angle + PI)
		scores[i] = 0.0


func _finish_round(reason: String) -> void:
	if not round_active:
		return

	round_active = false
	round_reason = reason

	for i in range(walkers.size()):
		var walker = walkers[i]
		if walker == null:
			continue
		scores[i] = walker.get_score(round_time_limit)
		if winner_index >= 0:
			if i == winner_index:
				scores[i] += winner_bonus
			else:
				scores[i] += participant_bonus

	best_score = maxf(best_score, scores.max() if not scores.is_empty() else 0.0)
	_update_hud()

	_evolve_policies()
	call_deferred("_start_round")


func _evolve_policies() -> void:
	generation += 1
	var order := _score_order_desc(scores)
	if order.is_empty():
		return

	var elite_a: PackedFloat32Array = PackedFloat32Array(policies[order[0]])
	var elite_b: PackedFloat32Array = PackedFloat32Array(elite_a)
	if order.size() > 1:
		elite_b = PackedFloat32Array(policies[order[1]])

	var next_policies: Array = []
	next_policies.resize(agent_count)
	next_policies[0] = PackedFloat32Array(elite_a)
	if agent_count > 1:
		next_policies[1] = PackedFloat32Array(elite_b)

	for i in range(2, agent_count):
		var parent: PackedFloat32Array = elite_a if i % 2 == 0 else elite_b
		next_policies[i] = _mutate_policy(parent, mutation_sigma, mutation_probability)

	policies = next_policies


func _score_order_desc(values: Array) -> Array:
	var indices: Array = []
	for i in range(values.size()):
		indices.append(i)
	indices.sort_custom(func(a, b): return float(values[a]) > float(values[b]))
	return indices


func _mutate_policy(parent: PackedFloat32Array, sigma: float, mutate_prob: float) -> PackedFloat32Array:
	var child := PackedFloat32Array(parent)
	for i in range(child.size()):
		if randf() < mutate_prob:
			child[i] += randfn(0.0, sigma)
	return child


func _move_target_to_random_position() -> void:
	if target_node == null:
		return

	var angle := randf_range(0.0, TAU)
	var radius := randf_range(3.5, arena_radius - 2.5)
	target_position = Vector3(cos(angle) * radius, 1.15, sin(angle) * radius)
	target_node.global_position = target_position


func _update_camera(delta: float) -> void:
	var focus := target_position
	if not walkers.is_empty():
		var avg_pos := Vector3.ZERO
		for walker_variant in walkers:
			var walker = walker_variant
			if walker == null:
				continue
			var status: Dictionary = walker.get_status(target_position)
			avg_pos += Vector3(walker.global_position.x, float(status.get("height", walker.global_position.y)), walker.global_position.z)
		avg_pos /= float(max(walkers.size(), 1))
		focus = avg_pos

	var desired := focus + Vector3(0.0, 9.0, 15.0)
	camera_rig.global_position = camera_rig.global_position.lerp(desired, clampf(delta * 2.0, 0.0, 1.0))
	camera_rig.look_at(focus + Vector3(0.0, 1.2, 0.0), Vector3.UP)


func _update_hud() -> void:
	var score_text := ""
	var order := _score_order_desc(scores)
	for rank in range(order.size()):
		var index: int = int(order[rank])
		var walker = walkers[index]
		if walker == null:
			continue
		var status: Dictionary = walker.get_status(target_position)
		score_text += "%d) IA-%d | score=%s | dist=%s | altura=%s | %s\n" % [
			rank + 1,
			index,
			String.num(float(scores[index]), 2),
			String.num(float(status.get("distance", 0.0)), 2),
			String.num(float(status.get("height", 0.0)), 2),
			"viva" if bool(status.get("alive", false)) else "fora"
		]

	info_label.text = "Geracao: %d\nRodada: %s\nTempo: %s / %s\nAlvo: (%.1f, %.1f)\nMelhor score: %s\n" % [
		generation,
		round_reason,
		String.num(round_elapsed, 2),
		String.num(round_time_limit, 2),
		target_position.x,
		target_position.z,
		String.num(best_score, 2)
	]
	ranking_label.text = score_text.strip_edges()


func _all_walkers_inactive() -> bool:
	for walker_variant in walkers:
		var walker = walker_variant
		if walker != null and walker.is_in_play():
			return false
	return true


func _on_reset_pressed() -> void:
	for i in range(walkers.size()):
		var walker = walkers[i]
		if walker == null:
			continue
		walker.set_policy(policies[i])
	call_deferred("_start_round")


func _on_shuffle_pressed() -> void:
	for i in range(walkers.size()):
		var walker = walkers[i]
		if walker == null:
			continue
		walker.randomize_policy()
		policies[i] = walker.policy
	call_deferred("_start_round")


func _on_menu_pressed() -> void:
	var err := get_tree().change_scene_to_file("res://scenes/SceneMenu.tscn")
	if err != OK:
		push_warning("Nao foi possivel voltar ao menu.")