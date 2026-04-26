extends Node
class_name OnnxPolicyClient

signal server_started
signal server_stopped
signal server_error(message: String)

const ACTION_SIZE := 8
const MODEL_CANDIDATE_PATHS := [
	"res://ml/models/agent_policy.onnx",
	"res://models/agent_policy.onnx",
    "res://Models/agent_policy.onnx"
]

@export var python_command: String = "python"
@export var server_script_path: String = "res://ml/onnx_inference_server.py"
@export var model_path: String = "res://ml/models/agent_policy.onnx"
@export var host: String = "127.0.0.1"
@export var port: int = 8765

var _udp := PacketPeerUDP.new()
var _server_pid: int = -1
var _last_action: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
var _server_ready: bool = false


func _process(_delta: float) -> void:
	_poll_packets()


func start_server(new_model_path: String = "") -> bool:
	if new_model_path != "":
		model_path = new_model_path

	model_path = _resolve_model_path()

	stop_server()

	var script_abs: String = ProjectSettings.globalize_path(server_script_path)
	var model_abs: String = ProjectSettings.globalize_path(model_path)
	if not FileAccess.file_exists(script_abs):
		emit_signal("server_error", "Script de inferencia nao encontrado")
		return false

	if not FileAccess.file_exists(model_abs):
		emit_signal("server_error", "Modelo ONNX nao encontrado")
		return false

	var args := PackedStringArray([
		script_abs,
		"--model",
		model_abs,
		"--host",
		host,
		"--port",
		str(port)
	])

	_server_pid = OS.create_process(python_command, args, false)
	if _server_pid <= 0:
		_server_pid = -1
		emit_signal("server_error", "Nao foi possivel iniciar o servidor ONNX")
		return false

	_udp = PacketPeerUDP.new()
	var err: Error = _udp.connect_to_host(host, port)
	if err != OK:
		if _server_pid > 0:
			OS.kill(_server_pid)
		_server_pid = -1
		emit_signal("server_error", "Nao foi possivel conectar no servidor ONNX")
		return false

	_server_ready = true
	emit_signal("server_started")
	return true


func stop_server() -> void:
	if _server_pid > 0:
		OS.kill(_server_pid)

	_server_pid = -1
	_server_ready = false
	_last_action = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	emit_signal("server_stopped")


func restart_server(new_model_path: String = "") -> bool:
	return start_server(new_model_path)


func request_action(observation: PackedFloat32Array) -> PackedFloat32Array:
	if not _server_ready:
		return PackedFloat32Array(_last_action)

	var payload := {
		"obs": observation
	}
	var message := JSON.stringify(payload)
	_udp.put_packet(message.to_utf8_buffer())
	_poll_packets()

	return PackedFloat32Array(_last_action)


func is_server_ready() -> bool:
	return _server_ready


func get_model_path() -> String:
	return _resolve_model_path()


func _poll_packets() -> void:
	while _udp.get_available_packet_count() > 0:
		var packet: PackedByteArray = _udp.get_packet()
		var text: String = packet.get_string_from_utf8()
		var parser := JSON.new()
		if parser.parse(text) != OK:
			continue

		var payload: Variant = parser.data
		if typeof(payload) != TYPE_DICTIONARY:
			continue

		var data: Dictionary = payload
		if not data.has("action"):
			continue

		_last_action = _sanitize_action(data["action"])


func _sanitize_action(action_variant: Variant) -> PackedFloat32Array:
	var sanitized := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	if typeof(action_variant) != TYPE_ARRAY:
		return sanitized

	var action_array: Array = action_variant
	var count: int = mini(action_array.size(), ACTION_SIZE)
	for i in range(count):
		sanitized[i] = float(action_array[i])

	return sanitized


func _resolve_model_path() -> String:
	if _model_exists(model_path):
		return model_path

	for path in MODEL_CANDIDATE_PATHS:
		if _model_exists(path):
			return path

	return model_path


func _model_exists(path: String) -> bool:
	return FileAccess.file_exists(ProjectSettings.globalize_path(path))
