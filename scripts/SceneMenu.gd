extends Control

@onready var open_main_button: Button = $Center/Panel/VBox/OpenMainButton
@onready var open_fullbody_button: Button = $Center/Panel/VBox/OpenFullBodyButton
@onready var open_parkour_button: Button = $Center/Panel/VBox/OpenParkourButton
@onready var open_mimic_button: Button = $Center/Panel/VBox/OpenMimicButton
@onready var quit_button: Button = $Center/Panel/VBox/QuitButton
@onready var status_label: Label = $Center/Panel/VBox/StatusLabel

const MAIN_SCENE_PATH := "res://scenes/Main.tscn"
const FULLBODY_SCENE_PATH := "res://scenes/FullBodyLocomotion.tscn"
const PARKOUR_SCENE_PATH := "res://scenes/ParkourCompetition.tscn"
const MIMIC_SCENE_PATH := "res://scenes/MimicExperiment.tscn"


func _ready() -> void:
	open_main_button.pressed.connect(_on_open_main_pressed)
	open_fullbody_button.pressed.connect(_on_open_fullbody_pressed)
	open_parkour_button.pressed.connect(_on_open_parkour_pressed)
	open_mimic_button.pressed.connect(_on_open_mimic_pressed)
	quit_button.pressed.connect(_on_quit_pressed)
	status_label.text = "Escolha uma cena para iniciar"


func _on_open_main_pressed() -> void:
	_open_scene(MAIN_SCENE_PATH)


func _on_open_fullbody_pressed() -> void:
	_open_scene(FULLBODY_SCENE_PATH)


func _on_open_parkour_pressed() -> void:
	_open_scene(PARKOUR_SCENE_PATH)


func _on_open_mimic_pressed() -> void:
	_open_scene(MIMIC_SCENE_PATH)


func _on_quit_pressed() -> void:
	get_tree().quit()


func _open_scene(scene_path: String) -> void:
	if not ResourceLoader.exists(scene_path):
		status_label.text = "Cena nao encontrada: %s" % scene_path
		return

	var err := get_tree().change_scene_to_file(scene_path)
	if err != OK:
		status_label.text = "Falha ao abrir: %s" % scene_path
