extends Node3D

@onready var mimic_node = $Mimic

# Referências aos materiais
var head_material: Material
var limbs_material: Material
var eye_material: Material

# Referências ao ragdoll
var skeleton: Skeleton3D
var bone_simulator: PhysicalBoneSimulator3D
var ragdoll_enabled: bool = false

func _ready():
	# Carregar os materiais
	head_material = load("res://Models/Mimic/DefMimic/materials/mat_head.tres")
	limbs_material = load("res://Models/Mimic/DefMimic/materials/mat_limbs.tres")
	eye_material = load("res://Models/Mimic/DefMimic/materials/mat_eye.tres")
	
	# Aplicar materiais aos meshes
	if mimic_node:
		apply_materials_recursive(mimic_node)
		
		# Encontrar o esqueleto e criar ragdoll
		skeleton = find_skeleton(mimic_node)
		if skeleton:
			print("Skeleton found: ", skeleton.name)
			create_ragdoll()
			enable_ragdoll()  # Ativar ragdoll automaticamente
		else:
			push_error("Skeleton3D not found in Mimic model!")
	else:
		push_error("Mimic node not found!")

func find_skeleton(node: Node) -> Skeleton3D:
	if node is Skeleton3D:
		return node
	for child in node.get_children():
		var result = find_skeleton(child)
		if result:
			return result
	return null

func apply_materials_recursive(node: Node):
	if node is MeshInstance3D:
		var node_name_lower = node.name.to_lower()
		
		# Aplicar material baseado no nome do mesh
		if "head" in node_name_lower:
			node.material_override = head_material
		elif "limb" in node_name_lower or "arm" in node_name_lower or "leg" in node_name_lower or "hand" in node_name_lower or "foot" in node_name_lower:
			node.material_override = limbs_material
		elif "eye" in node_name_lower:
			node.material_override = eye_material
	
	# Recursivamente aplicar aos filhos
	for child in node.get_children():
		apply_materials_recursive(child)

func create_ragdoll():
	if not skeleton:
		return
	
	print("Creating ragdoll for skeleton with ", skeleton.get_bone_count(), " bones")
	
	# Criar PhysicalBoneSimulator3D como pai do Skeleton3D
	bone_simulator = PhysicalBoneSimulator3D.new()
	bone_simulator.name = "PhysicalBoneSimulator"
	
	# Reparentar o Skeleton3D para o PhysicalBoneSimulator3D
	var skeleton_parent = skeleton.get_parent()
	skeleton_parent.remove_child(skeleton)
	bone_simulator.add_child(skeleton)
	skeleton_parent.add_child(bone_simulator)
	
	# Lista de bones principais para criar PhysicalBone3D
	var main_bones = []
	for i in range(skeleton.get_bone_count()):
		var bone_name = skeleton.get_bone_name(i)
		print("Bone ", i, ": ", bone_name)
		main_bones.append(bone_name)
	
	# Criar PhysicalBone3D para cada bone principal
	for bone_name in main_bones:
		create_physical_bone(bone_name)

func create_physical_bone(bone_name: String):
	var bone_id = skeleton.find_bone(bone_name)
	if bone_id == -1:
		return
	
	var physical_bone = PhysicalBone3D.new()
	physical_bone.name = "PhysicalBone_" + bone_name
	physical_bone.bone_name = bone_name
	
	# Configurar massa baseada no tamanho do bone
	physical_bone.mass = 1.0
	physical_bone.friction = 0.8
	physical_bone.bounce = 0.1
	physical_bone.gravity_scale = 1.0
	
	# Adicionar como filho do PhysicalBoneSimulator3D (não do Skeleton3D)
	bone_simulator.add_child(physical_bone)
	
	# Criar colisor para o bone
	create_bone_collider(physical_bone, bone_name)
	
	print("Created PhysicalBone for: ", bone_name)

func create_bone_collider(physical_bone: PhysicalBone3D, bone_name: String):
	var collision_shape = CollisionShape3D.new()
	collision_shape.name = "CollisionShape"
	
	# Criar forma baseada no nome do bone
	var bone_name_lower = bone_name.to_lower()
	
	if "head" in bone_name_lower:
		var sphere = SphereShape3D.new()
		sphere.radius = 0.3
		collision_shape.shape = sphere
	elif "spine" in bone_name_lower or "torso" in bone_name_lower or "chest" in bone_name_lower:
		var capsule = CapsuleShape3D.new()
		capsule.radius = 0.25
		capsule.height = 0.5
		collision_shape.shape = capsule
	elif "arm" in bone_name_lower:
		var capsule = CapsuleShape3D.new()
		capsule.radius = 0.1
		capsule.height = 0.4
		collision_shape.shape = capsule
	elif "leg" in bone_name_lower:
		var capsule = CapsuleShape3D.new()
		capsule.radius = 0.12
		capsule.height = 0.5
		collision_shape.shape = capsule
	elif "hand" in bone_name_lower or "foot" in bone_name_lower:
		var box = BoxShape3D.new()
		box.size = Vector3(0.15, 0.1, 0.2)
		collision_shape.shape = box
	else:
		# Default: capsule pequeno
		var capsule = CapsuleShape3D.new()
		capsule.radius = 0.1
		capsule.height = 0.3
		collision_shape.shape = capsule
	
	physical_bone.add_child(collision_shape)

func enable_ragdoll():
	if bone_simulator:
		bone_simulator.physical_bones_start_simulation()
		ragdoll_enabled = true
		print("Ragdoll enabled")

func disable_ragdoll():
	if bone_simulator:
		bone_simulator.physical_bones_stop_simulation()
		ragdoll_enabled = false
		print("Ragdoll disabled")
