extends Node3D

@export var episode_time_limit: float = 45.0
@export var capture_distance: float = 1.4
@export var min_spawn_radius: float = 9.0
@export var max_spawn_radius: float = 14.0
@export var arena_radius: float = 30.0
@export var ragdoll_seconds_on_failure: float = 1.1
@export var auto_start_onnx_server: bool = true
@export var use_onnx_policy: bool = true

@onready var agent: HumanoidAgent = $HumanoidAgent
@onready var pursuer: PursuerCapsule = $Pursuer
@onready var hud: SurvivalHUD = $CanvasLayer/HUD
@onready var camera_rig: Node3D = $CameraRig
@onready var policy_client: OnnxPolicyClient = $OnnxPolicyClient
@onready var training_manager: TrainingManager = $TrainingManager

var total_episodes: int = 0
var episode_elapsed: float = 0.0
var best_survival: float = 0.0
var last_episode_duration: float = 0.0
var last_result: String = "Aguardando"
var current_action: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
var current_distance: float = 0.0
var _episode_end_pending: bool = false
var _episode_end_timer: float = 0.0


func _ready() -> void:
	randomize()

	pursuer.set_target(agent)

	hud.start_training_requested.connect(_on_start_training_requested)
	hud.stop_training_requested.connect(_on_stop_training_requested)
	hud.restart_episode_requested.connect(_on_restart_episode_requested)
	hud.onnx_usage_toggled.connect(_on_onnx_toggled)

	training_manager.training_started.connect(_on_training_started)
	training_manager.training_stopped.connect(_on_training_stopped)
	training_manager.training_finished.connect(_on_training_finished)

	policy_client.server_error.connect(_on_policy_server_error)

	_reset_episode()
	hud.set_system_message("Simulacao pronta")

	if auto_start_onnx_server and use_onnx_policy:
		_start_policy_server()


func _physics_process(delta: float) -> void:
	if _episode_end_pending:
		_episode_end_timer -= delta
		if _episode_end_timer <= 0.0:
			_end_episode_now()
		_update_camera(delta)
		_update_hud()
		return

	episode_elapsed += delta

	var observation := agent.get_observation(
		pursuer.global_position,
		pursuer.linear_velocity,
		episode_elapsed,
		episode_time_limit
	)

	current_action = _choose_action(observation)
	agent.apply_action(current_action)

	current_distance = agent.global_position.distance_to(pursuer.global_position)

	if _should_end_episode():
		_end_episode()

	_update_camera(delta)
	_update_hud()


func _choose_action(observation: PackedFloat32Array) -> PackedFloat32Array:
	if use_onnx_policy and policy_client.is_server_ready():
		return policy_client.request_action(observation)

	return _heuristic_action(observation)


func _heuristic_action(observation: PackedFloat32Array) -> PackedFloat32Array:
	var dx: float = _obs_value(observation, 0)
	var dz: float = _obs_value(observation, 2)
	var distance: float = maxf(_obs_value(observation, 9), 0.001)
	var on_floor: bool = _obs_value(observation, 10) > 0.5

	var time := Time.get_ticks_msec() * 0.001
	var desired_dir := Vector2(cos(time * 0.35), sin(time * 0.35)).normalized()

	var threat_dir := Vector2(dx, dz)
	if threat_dir.length() > 0.001:
		threat_dir = threat_dir.normalized()

	# Prioriza locomocao em um trajeto circular; apenas faz evasao quando perseguido de perto.
	var movement_dir := desired_dir
	if distance < 4.0:
		movement_dir = (desired_dir - threat_dir * 0.6).normalized()

	var jump := 1.0 if distance < 3.8 and on_floor else -1.0
	var gait := sin(time * 6.2)

	return PackedFloat32Array([
		movement_dir.x,
		movement_dir.y,
		jump,
		sin(time * 0.75),
		gait,
		-gait,
		-gait,
		gait
	])


func _should_end_episode() -> bool:
	if _planar_length(agent.global_position) > arena_radius + 1.5:
		last_result = "Saiu do limite da arena"
		return true

	if current_distance <= capture_distance:
		last_result = "Capturado pela capsula"
		return true

	if agent.global_position.y < -3.0:
		last_result = "Queda fora da arena"
		return true

	if episode_elapsed >= episode_time_limit:
		last_result = "Sobreviveu ao limite"
		return true

	return false


func _end_episode() -> void:
	_episode_end_pending = true
	_episode_end_timer = ragdoll_seconds_on_failure
	agent.set_ragdoll_enabled(true)


func _end_episode_now() -> void:
	last_episode_duration = episode_elapsed
	best_survival = max(best_survival, last_episode_duration)
	total_episodes += 1
	_episode_end_pending = false
	_episode_end_timer = 0.0
	_reset_episode()


func _reset_episode() -> void:
	episode_elapsed = 0.0
	last_result = "Episodio em andamento"
	current_action = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	agent.set_ragdoll_enabled(false)

	var agent_spawn := Vector3(0.0, 1.2, 0.0)
	var angle := randf_range(0.0, TAU)
	var radius := randf_range(min_spawn_radius, max_spawn_radius)
	var pursuer_spawn := Vector3(cos(angle) * radius, 1.0, sin(angle) * radius)

	agent.reset_agent(agent_spawn)
	pursuer.reset_pursuer(pursuer_spawn)


func _update_camera(delta: float) -> void:
	var desired_position := agent.global_position + Vector3(0.0, 7.0, 11.0)
	camera_rig.global_position = camera_rig.global_position.lerp(desired_position, delta * 2.5)
	camera_rig.look_at(agent.global_position + Vector3(0.0, 1.5, 0.0), Vector3.UP)


func _update_hud() -> void:
	var game_data := {
		"episode": total_episodes + 1,
		"elapsed": episode_elapsed,
		"best_survival": best_survival,
		"last_duration": last_episode_duration,
		"distance": current_distance,
		"policy_mode": "ONNX" if use_onnx_policy else "Heuristica",
		"onnx_server": policy_client.is_server_ready(),
		"result": last_result
	}

	var agent_data := agent.get_agent_status(current_distance)
	var pursuer_data := pursuer.get_status()
	var training_status := training_manager.get_status()

	hud.update_game_stats(game_data)
	hud.update_agent_stats(agent_data, pursuer_data, current_action)
	hud.update_training_stats(
		training_status,
		training_manager.is_training(),
		policy_client.get_model_path(),
		policy_client.is_server_ready()
	)


func _on_start_training_requested(config: Dictionary) -> void:
	var started := training_manager.start_training(config)
	if started:
		hud.set_system_message("Treino iniciado")
	else:
		hud.set_system_message("Falha ao iniciar treino")


func _on_stop_training_requested() -> void:
	training_manager.stop_training()
	hud.set_system_message("Treino interrompido")


func _on_restart_episode_requested() -> void:
	last_result = "Reinicio manual"
	_end_episode()


func _on_onnx_toggled(enabled: bool) -> void:
	use_onnx_policy = enabled
	if use_onnx_policy:
		_start_policy_server()
	else:
		policy_client.stop_server()
		hud.set_system_message("Modo heuristico ativo")


func _on_training_started() -> void:
	hud.set_system_message("Treinamento em execucao")


func _on_training_stopped() -> void:
	hud.set_system_message("Treinamento parado")


func _on_training_finished(state: String) -> void:
	hud.set_system_message("Treinamento finalizado: %s" % state)
	if use_onnx_policy:
		_start_policy_server()


func _on_policy_server_error(message: String) -> void:
	hud.set_system_message("ONNX: %s" % message)


func _start_policy_server() -> void:
	var model_abs := ProjectSettings.globalize_path(policy_client.get_model_path())
	if not FileAccess.file_exists(model_abs):
		hud.set_system_message("Modelo ONNX ainda nao existe")
		return

	if policy_client.restart_server(policy_client.get_model_path()):
		hud.set_system_message("Servidor ONNX ativo")
	else:
		hud.set_system_message("Nao foi possivel iniciar ONNX")


func _obs_value(observation: PackedFloat32Array, index: int) -> float:
	if index < observation.size():
		return observation[index]
	return 0.0


func _planar_length(pos: Vector3) -> float:
	return Vector2(pos.x, pos.z).length()
