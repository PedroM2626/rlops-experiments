extends Control
class_name SurvivalHUD

signal start_training_requested(config: Dictionary)
signal stop_training_requested
signal restart_episode_requested
signal onnx_usage_toggled(enabled: bool)

@onready var game_stats_label: RichTextLabel = $PanelsContainer/GamePanel/Content/GameVBox/GameStats
@onready var agent_stats_label: RichTextLabel = $PanelsContainer/AgentPanel/Content/AgentVBox/AgentStats
@onready var training_stats_label: RichTextLabel = $PanelsContainer/TrainingPanel/Content/TrainingVBox/TrainingStats

@onready var onnx_check: CheckButton = $ControlsPanel/Content/ControlsVBox/UseOnnxCheck
@onready var reuse_check: CheckButton = $ControlsPanel/Content/ControlsVBox/ReuseCheckpointCheck
@onready var generations_spin: SpinBox = $ControlsPanel/Content/ControlsVBox/GenerationsRow/GenerationsSpin
@onready var population_spin: SpinBox = $ControlsPanel/Content/ControlsVBox/PopulationRow/PopulationSpin
@onready var episode_spin: SpinBox = $ControlsPanel/Content/ControlsVBox/EpisodeRow/EpisodeSpin
@onready var seed_spin: SpinBox = $ControlsPanel/Content/ControlsVBox/SeedRow/SeedSpin
@onready var start_training_button: Button = $ControlsPanel/Content/ControlsVBox/StartTrainingButton
@onready var stop_training_button: Button = $ControlsPanel/Content/ControlsVBox/StopTrainingButton
@onready var restart_episode_button: Button = $ControlsPanel/Content/ControlsVBox/RestartEpisodeButton
@onready var model_label: Label = $ControlsPanel/Content/ControlsVBox/ModelPathLabel
@onready var system_message_label: Label = $ControlsPanel/Content/ControlsVBox/SystemMessageLabel


func _ready() -> void:
    start_training_button.pressed.connect(_on_start_training_pressed)
    stop_training_button.pressed.connect(_on_stop_training_pressed)
    restart_episode_button.pressed.connect(_on_restart_episode_pressed)
    onnx_check.toggled.connect(_on_onnx_toggled)


func update_game_stats(game_data: Dictionary) -> void:
    var text := ""
    text += "Episodio atual: %d\n" % int(game_data.get("episode", 1))
    text += "Tempo no episodio: %s s\n" % _fmt(game_data.get("elapsed", 0.0), 2)
    text += "Melhor sobrevivencia: %s s\n" % _fmt(game_data.get("best_survival", 0.0), 2)
    text += "Ultimo episodio: %s s\n" % _fmt(game_data.get("last_duration", 0.0), 2)
    text += "Distancia perseguidor: %s m\n" % _fmt(game_data.get("distance", 0.0), 2)
    text += "Modo de politica: %s\n" % str(game_data.get("policy_mode", "Heuristica"))
    text += "Servidor ONNX: %s\n" % ("online" if bool(game_data.get("onnx_server", false)) else "offline")
    text += "Estado episodio: %s" % str(game_data.get("result", "Rodando"))
    game_stats_label.text = text


func update_agent_stats(agent_data: Dictionary, pursuer_data: Dictionary, action: PackedFloat32Array) -> void:
    var text := ""
    text += "Velocidade agente: %s m/s\n" % _fmt(agent_data.get("speed", 0.0), 2)
    text += "Velocidade vertical: %s m/s\n" % _fmt(agent_data.get("vertical_speed", 0.0), 2)
    text += "Altura: %s m\n" % _fmt(agent_data.get("height", 0.0), 2)
    text += "No chao: %s\n" % ("sim" if bool(agent_data.get("on_floor", false)) else "nao")
    text += "Velocidade perseguidor: %s m/s\n" % _fmt(pursuer_data.get("speed", 0.0), 2)
    text += "Acao atual [movX movZ jump]: %s %s %s\n" % [
        _fmt(_action_value(action, 0), 2),
        _fmt(_action_value(action, 1), 2),
        _fmt(_action_value(action, 2), 2)
    ]
    text += "Acao membros [cab brE brD peE peD]: %s %s %s %s %s" % [
        _fmt(_action_value(action, 3), 2),
        _fmt(_action_value(action, 4), 2),
        _fmt(_action_value(action, 5), 2),
        _fmt(_action_value(action, 6), 2),
        _fmt(_action_value(action, 7), 2)
    ]
    agent_stats_label.text = text


func update_training_stats(training_data: Dictionary, training_active: bool, model_path: String, onnx_ready: bool) -> void:
    var state := str(training_data.get("state", "idle"))
    var text := ""
    text += "Treino ativo: %s\n" % ("sim" if training_active else "nao")
    text += "Estado: %s\n" % state
    text += "Geracao: %d\n" % int(training_data.get("generation", 0))
    text += "Melhor fitness: %s\n" % _fmt(training_data.get("best_reward", 0.0), 3)
    text += "Reward atual: %s\n" % _fmt(training_data.get("current_reward", 0.0), 3)
    text += "Populacao: %d\n" % int(training_data.get("population", 0))
    text += "Tempo episodio treino: %s s\n" % _fmt(training_data.get("episode_seconds", 0.0), 2)
    text += "Checkpoint reusado: %s\n" % ("sim" if bool(training_data.get("reused_checkpoint", false)) else "nao")
    text += "ONNX salvo: %s\n" % str(training_data.get("onnx_path", "pendente"))
    text += "Inferencia pronta: %s" % ("sim" if onnx_ready else "nao")

    training_stats_label.text = text
    model_label.text = "Modelo ativo: %s" % model_path


func set_system_message(message: String) -> void:
    system_message_label.text = "Status: %s" % message


func is_onnx_enabled() -> bool:
    return onnx_check.button_pressed


func _on_start_training_pressed() -> void:
    var config := {
        "generations": int(generations_spin.value),
        "population": int(population_spin.value),
        "episode_seconds": float(episode_spin.value),
        "seed": int(seed_spin.value),
        "reuse_checkpoint": reuse_check.button_pressed
    }
    emit_signal("start_training_requested", config)


func _on_stop_training_pressed() -> void:
    emit_signal("stop_training_requested")


func _on_restart_episode_pressed() -> void:
    emit_signal("restart_episode_requested")


func _on_onnx_toggled(enabled: bool) -> void:
    emit_signal("onnx_usage_toggled", enabled)


func _fmt(value: Variant, digits: int) -> String:
    return String.num(float(value), digits)


func _action_value(action: PackedFloat32Array, index: int) -> float:
    if index < action.size():
        return action[index]
    return 0.0
