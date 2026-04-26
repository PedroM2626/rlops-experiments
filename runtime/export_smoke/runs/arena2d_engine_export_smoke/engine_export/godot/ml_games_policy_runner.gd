class_name MLGamesPolicyRunner
extends RefCounted

const OBSERVATION_SIZE := 10
const ACTION_SIZE := 2

var backend: Callable

func _init(inference_backend: Callable) -> void:
	backend = inference_backend

func predict(observation: PackedFloat32Array) -> PackedFloat32Array:
	assert(observation.size() == OBSERVATION_SIZE)
	var action = backend.call(observation)
	assert(action.size() == ACTION_SIZE)
	return action
