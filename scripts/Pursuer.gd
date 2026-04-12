extends RigidBody3D
class_name PursuerCapsule

@export var chase_force: float = 45.0
@export var max_speed: float = 10.0
@export var vertical_stabilizer: float = 20.0
@export var hover_height: float = 1.0
@export var drag: float = 0.14

var target: Node3D = null
var recent_distance: float = INF


func _ready() -> void:
    can_sleep = false
    contact_monitor = true
    max_contacts_reported = 8
    axis_lock_angular_x = true
    axis_lock_angular_z = true


func _physics_process(_delta: float) -> void:
    if target == null:
        return

    var target_position := target.global_position + Vector3.UP * hover_height
    var offset := target_position - global_position
    recent_distance = offset.length()

    if offset.length_squared() > 0.0001:
        apply_central_force(offset.normalized() * chase_force)

    apply_central_force(-linear_velocity * drag)
    linear_velocity = linear_velocity.limit_length(max_speed)

    var vertical_error := target_position.y - global_position.y
    apply_central_force(Vector3.UP * vertical_error * vertical_stabilizer)


func set_target(new_target: Node3D) -> void:
    target = new_target


func reset_pursuer(spawn_position: Vector3) -> void:
    global_position = spawn_position
    linear_velocity = Vector3.ZERO
    angular_velocity = Vector3.ZERO
    sleeping = false


func get_status() -> Dictionary:
    return {
        "speed": linear_velocity.length(),
        "distance": recent_distance,
        "position": global_position
    }
