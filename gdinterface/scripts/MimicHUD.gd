extends Control

@onready var health_bar: ProgressBar = $HealthBar
@onready var health_label: Label = $HealthLabel

var player: CharacterBody3D

func _ready():
	# Encontrar o player na cena
	player = get_tree().get_first_node_in_group("player")
	if not player:
		player = get_node("/root/MimicExperiment/Player")
	
	if player:
		player.health_changed.connect(_on_health_changed)
		player.player_died.connect(_on_player_died)
		health_bar.max_value = player.max_health
		health_bar.value = player.current_health
		health_label.text = "Health: %.0f/%.0f" % [player.current_health, player.max_health]

func _on_health_changed(new_health: float):
	if health_bar:
		health_bar.value = new_health
		health_label.text = "Health: %.0f/%.0f" % [new_health, health_bar.max_value]

func _on_player_died():
	health_label.text = "YOU DIED"
	health_bar.value = 0
