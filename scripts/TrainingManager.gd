extends Node
class_name TrainingManager

signal training_started
signal training_stopped
signal training_finished(state: String)

@export var python_command: String = "python"
@export var trainer_script_path: String = "res://ml/train_agent.py"
@export var model_directory: String = "res://ml/models"
@export var status_file: String = "res://ml/models/training_status.json"

var _training_pid: int = -1
var _training_active: bool = false
var _cached_status: Dictionary = {
    "state": "idle"
}
var _poll_accumulator: float = 0.0


func _process(delta: float) -> void:
    _poll_accumulator += delta
    if _poll_accumulator < 0.5:
        return

    _poll_accumulator = 0.0
    _cached_status = _read_status_file()

    if _training_active:
        var state := str(_cached_status.get("state", "running"))
        if state == "finished" or state == "error" or state == "interrupted":
            _training_active = false
            _training_pid = -1
            emit_signal("training_finished", state)


func start_training(config: Dictionary) -> bool:
    if _training_active:
        return false

    var script_abs := ProjectSettings.globalize_path(trainer_script_path)
    if not FileAccess.file_exists(script_abs):
        return false

    var model_abs := ProjectSettings.globalize_path(model_directory)
    var generations := int(config.get("generations", 40))
    var population := int(config.get("population", 24))
    var episode_seconds := float(config.get("episode_seconds", 45.0))
    var reuse_checkpoint := bool(config.get("reuse_checkpoint", true))
    var seed := int(config.get("seed", 7))

    var args := PackedStringArray([
        script_abs,
        "--model-dir",
        model_abs,
        "--generations",
        str(generations),
        "--population",
        str(population),
        "--episode-seconds",
        str(episode_seconds),
        "--reuse",
        "true" if reuse_checkpoint else "false",
        "--seed",
        str(seed)
    ])

    _training_pid = OS.create_process(python_command, args, false)
    if _training_pid <= 0:
        _training_pid = -1
        _training_active = false
        return false

    _training_active = true
    _cached_status = {
        "state": "starting"
    }
    emit_signal("training_started")
    return true


func stop_training() -> void:
    if _training_pid > 0:
        OS.kill(_training_pid)

    _training_pid = -1
    _training_active = false
    _cached_status = {
        "state": "interrupted"
    }
    emit_signal("training_stopped")


func is_training() -> bool:
    return _training_active


func get_status() -> Dictionary:
    if _training_active:
        return _cached_status

    var current := _read_status_file()
    if current.is_empty():
        return _cached_status
    return current


func _read_status_file() -> Dictionary:
    var status_abs := ProjectSettings.globalize_path(status_file)
    if not FileAccess.file_exists(status_abs):
        return {}

    var file := FileAccess.open(status_abs, FileAccess.READ)
    if file == null:
        return {}

    var content := file.get_as_text()
    if content.is_empty():
        return {}

    var parser := JSON.new()
    if parser.parse(content) != OK:
        return {}

    if typeof(parser.data) != TYPE_DICTIONARY:
        return {}

    return parser.data
