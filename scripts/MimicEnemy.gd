extends Node3D

enum State {
	PATROL,
	CHASE_SPEED,
	CHASE_STRENGTH
}

enum ChaseMode {
	SPEED,
	STRENGTH
}

# Referências
@onready var navigation_agent: NavigationAgent3D = $NavigationAgent3D
@onready var vision_area: Area3D = $Area3D
@onready var vision_shape: CollisionShape3D = $Area3D/CollisionShape3D
@onready var light: OmniLight3D = $OmniLight3D
@onready var mimic_node = $"."

# Configurações de campo de visão
@export var vision_range: float = 20.0
@export var vision_angle: float = 90.0
@export var patrol_speed: float = 2.0
@export var chase_speed_speed: float = 4.0
@export var chase_strength_speed: float = 1.5

# Configurações de combate
@export var speed_mode_damage: float = 5.0
@export var strength_mode_damage: float = 25.0
@export var max_stamina: float = 100.0
@export var stamina_drain_speed: float = 10.0
@export var stamina_drain_strength: float = 20.0
@export var stamina_regen: float = 5.0
@export var attack_cooldown: float = 2.0
@export var mode_switch_cooldown: float = 5.0

# Variáveis de estado
var current_state: State = State.PATROL
var current_chase_mode: ChaseMode = ChaseMode.SPEED
var current_stamina: float = max_stamina
var attack_timer: float = 0.0
var mode_switch_timer: float = 0.0
var player: CharacterBody3D = null
var patrol_points: Array[Vector3] = []
var current_patrol_index: int = 0
var skeleton: Skeleton3D
var animation_player: AnimationPlayer
var animation_tree: AnimationTree

# Materiais
var head_material: Material
var limbs_material: Material
var eye_material: Material

func _ready():
	# Configurar NavigationAgent
	navigation_agent.velocity_computed.connect(_on_velocity_computed)
	
	# Configurar campo de visão
	setup_vision()
	
	# Carregar materiais
	head_material = load("res://Models/Mimic/DefMimic/materials/mat_head.tres")
	limbs_material = load("res://Models/Mimic/DefMimic/materials/mat_limbs.tres")
	eye_material = load("res://Models/Mimic/DefMimic/materials/mat_eye.tres")
	
	# Aplicar materiais
	if mimic_node:
		apply_materials_recursive(mimic_node)
	
	# Encontrar skeleton, animation player e animation tree
	skeleton = find_skeleton(mimic_node)
	animation_player = find_animation_player(mimic_node)
	animation_tree = find_animation_tree(mimic_node)
	
	print("Skeleton found: ", skeleton != null)
	print("AnimationPlayer found: ", animation_player != null)
	print("AnimationTree found: ", animation_tree != null)
	
	if animation_player:
		print("Available animations: ", animation_player.get_animation_list())
	
	if animation_tree:
		animation_tree.active = true
	
	# Gerar pontos de patrulha
	generate_patrol_points()
	
	# Configurar luz inicial
	update_light_color()

func _physics_process(delta):
	match current_state:
		State.PATROL:
			patrol_behavior(delta)
		State.CHASE_SPEED, State.CHASE_STRENGTH:
			chase_behavior(delta)
	
	# Atualizar timers
	if attack_timer > 0:
		attack_timer -= delta
	if mode_switch_timer > 0:
		mode_switch_timer -= delta
	
	# Regenerar stamina quando não está em perseguição
	if current_state == State.PATROL:
		current_stamina = min(current_stamina + stamina_regen * delta, max_stamina)
	
	# Verificar campo de visão
	check_vision()

func setup_vision():
	# Configurar Area3D para detecção de jogador
	vision_area.monitoring = true
	
	# Configurar CollisionShape3D como esfera para o campo de visão
	var sphere_shape = SphereShape3D.new()
	sphere_shape.radius = vision_range
	vision_shape.shape = sphere_shape
	
	# Conectar sinal de corpo entrando na área
	vision_area.body_entered.connect(_on_body_entered_vision)
	vision_area.body_exited.connect(_on_body_exited_vision)

func generate_patrol_points():
	# Gerar pontos de patrulha aleatórios ao redor da posição inicial
	var base_position = global_position
	for i in range(5):
		var angle = (PI * 2 / 5) * i
		var distance = 10.0
		var offset = Vector3(cos(angle) * distance, 0, sin(angle) * distance)
		patrol_points.append(base_position + offset)

func patrol_behavior(delta):
	if patrol_points.is_empty():
		return
	
	var target_point = patrol_points[current_patrol_index]
	navigation_agent.set_target_position(target_point)
	
	if navigation_agent.is_navigation_finished():
		current_patrol_index = (current_patrol_index + 1) % patrol_points.size()
	
	move_towards_target(patrol_speed, delta)
	play_animation("Mimic|Fastwalk")

func chase_behavior(delta):
	var chase_speed = chase_speed_speed if current_chase_mode == ChaseMode.SPEED else chase_strength_speed
	
	if player:
		navigation_agent.set_target_position(player.global_position)
		
		# Verificar se pode atacar
		var distance_to_player = global_position.distance_to(player.global_position)
		if distance_to_player < 2.5 and attack_timer <= 0:
			attack_player()
		
		# Gerenciar modo de perseguição baseado em stamina
		if mode_switch_timer <= 0:
			if current_chase_mode == ChaseMode.SPEED and current_stamina < 15.0:
				switch_chase_mode(ChaseMode.STRENGTH)
			elif current_chase_mode == ChaseMode.STRENGTH and current_stamina > 60.0:
				switch_chase_mode(ChaseMode.SPEED)
		
		# Drenar stamina durante perseguição
		var stamina_drain = stamina_drain_speed if current_chase_mode == ChaseMode.SPEED else stamina_drain_strength
		current_stamina = max(current_stamina - stamina_drain * delta, 0)
		
		# Escolher animação baseada no modo e stamina
		if current_chase_mode == ChaseMode.SPEED:
			if current_stamina > 40.0:
				play_animation("Mimic|Sprint_400")
			else:
				play_animation("Mimic|Fastwalk_225")
		else:
			play_animation("Mimic|Fastwalk_225")
	
	move_towards_target(chase_speed, delta)

func move_towards_target(speed: float, delta: float):
	if navigation_agent.is_navigation_finished():
		return
	
	var next_position = navigation_agent.get_next_path_position()
	var direction = global_position.direction_to(next_position)
	var velocity = direction * speed
	
	# Rotacionar para olhar na direção do movimento
	if direction.length() > 0.01:
		look_at(global_position - direction, Vector3.UP)
	
	if navigation_agent.avoidance_enabled:
		navigation_agent.set_velocity(velocity)
	else:
		_on_velocity_computed(velocity)

func _on_velocity_computed(safe_velocity: Vector3):
	global_position += safe_velocity * get_physics_process_delta_time()

func check_vision():
	if not player:
		return
	
	var distance_to_player = global_position.distance_to(player.global_position)
	
	# Se o player está dentro do alcance da área de visão, detecta automaticamente
	if distance_to_player < vision_range:
		# Jogador detectado, mudar para estado de perseguição
		if current_state == State.PATROL:
			current_state = State.CHASE_SPEED
			current_chase_mode = ChaseMode.SPEED
			update_light_color()
	else:
		# Jogador fora do alcance
		if current_state != State.PATROL:
			current_state = State.PATROL
			update_light_color()

func switch_chase_mode(new_mode: ChaseMode):
	current_chase_mode = new_mode
	mode_switch_timer = mode_switch_cooldown
	update_light_color()

func attack_player():
	if not player:
		return
	
	var damage = speed_mode_damage if current_chase_mode == ChaseMode.SPEED else strength_mode_damage
	
	# Enviar sinal de dano ao player se tiver o método
	if player.has_method("take_damage"):
		player.take_damage(damage)
	
	print("Attacking player with ", damage, " damage in ", current_chase_mode, " mode")
	
	attack_timer = attack_cooldown

func update_light_color():
	match current_state:
		State.PATROL:
			light.light_color = Color.WHITE
		State.CHASE_SPEED:
			light.light_color = Color.ORANGE
		State.CHASE_STRENGTH:
			light.light_color = Color.RED

func play_animation(anim_name: String):
	# Priorizar AnimationTree se disponível
	if animation_tree:
		# Usar AnimationTree para controlar animações
		var parameters = animation_tree.get("parameters")
		if parameters.has("blend_position"):
			# Se for um BlendSpace2D, pode precisar configurar blend_position
			pass
		# Para AnimationTree, geralmente usamos parâmetros em vez de play direto
		print("Using AnimationTree, but direct play not supported. Consider using parameters.")
		return
	
	if not animation_player:
		return
	
	# Verificar se a animação existe
	if not animation_player.has_animation(anim_name):
		print("Animation not found: ", anim_name)
		# Tentar usar animação idle como fallback
		if animation_player.has_animation("Mimic|Idle"):
			anim_name = "Mimic|Idle"
		else:
			return
	
	if animation_player.current_animation != anim_name:
		animation_player.play(anim_name)

func find_skeleton(node: Node) -> Skeleton3D:
	if node is Skeleton3D:
		return node
	for child in node.get_children():
		var result = find_skeleton(child)
		if result:
			return result
	return null

func find_animation_player(node: Node) -> AnimationPlayer:
	if node is AnimationPlayer:
		return node
	for child in node.get_children():
		var result = find_animation_player(child)
		if result:
			return result
	return null

func find_animation_tree(node: Node) -> AnimationTree:
	if node is AnimationTree:
		return node
	for child in node.get_children():
		var result = find_animation_tree(child)
		if result:
			return result
	return null

func apply_materials_recursive(node: Node):
	if node is MeshInstance3D:
		var node_name_lower = node.name.to_lower()
		
		if "head" in node_name_lower:
			node.material_override = head_material
		elif "limb" in node_name_lower or "arm" in node_name_lower or "leg" in node_name_lower or "hand" in node_name_lower or "foot" in node_name_lower:
			node.material_override = limbs_material
		elif "eye" in node_name_lower:
			node.material_override = eye_material
	
	for child in node.get_children():
		apply_materials_recursive(child)

func _on_body_entered_vision(body: Node3D):
	if body is CharacterBody3D and body.name == "Player":
		player = body

func _on_body_exited_vision(body: Node3D):
	if body == player:
		player = null
