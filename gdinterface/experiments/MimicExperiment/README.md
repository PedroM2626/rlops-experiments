# Mimic Experiment

Experimento de IA para o modelo Mimic com sistema de patrulhamento e perseguição.

## Características

### Player Controlável
- Movimento WASD
- Pulo com Espaço
- Controle de câmera com mouse
- Captura do mouse automática (ESC para liberar)

### IA do Mimic

#### Sistema de Estados (State Machine)
- **PATROL**: Patrulhamento automático por pontos predefinidos
- **CHASE_SPEED**: Perseguição em modo velocidade
- **CHASE_STRENGTH**: Perseguição em modo força

#### Campo de Visão
- Raio de visão configurável (padrão: 20.0 unidades)
- Ângulo de visão configurável (padrão: 90 graus)
- Detecção via RayCast para verificar linha de visão
- Area3D para detecção de entrada/saída do jogador

#### Modos de Perseguição

**Modo Velocidade (SPEED)**
- Velocidade de movimento rápida (4.0 unidades/s)
- Dano baixo (5.0)
- Drena 10.0 de stamina por segundo
- Animação: sprint_400 (com stamina) ou fastwalk_225 (sem stamina)
- Luz: Laranja

**Modo Força (STRENGTH)**
- Velocidade de movimento lenta (1.5 unidades/s)
- Dano alto (25.0)
- Drena 20.0 de stamina por segundo
- Animação: fastwalk_225
- Luz: Vermelha

#### Sistema de Stamina
- Stamina máxima: 100.0
- Regeneração: 5.0 por segundo (apenas em patrulha)
- Drenagem durante perseguição
- Troca automática de modo baseada em stamina

#### Sistema de Cooldown
- Cooldown de ataque: 2.0 segundos
- Cooldown de troca de modo: 5.0 segundos

#### Luz Dinâmica
- Branco: Patrulhamento
- Laranja: Perseguição modo velocidade
- Vermelho: Perseguição modo força

#### Animações
- idle: Parado
- fastwalk: Patrulhamento
- fastwalk_225: Perseguição sem stamina
- sprint_400: Perseguição com stamina

## Configuração

### Cena Principal
Arquivo: `scenes/MimicExperiment.tscn`

### Scripts
- `scripts/Player.gd`: Script do player controlável
- `scripts/MimicEnemy.gd`: Script da IA do Mimic

### Parâmetros Configuráveis (no Editor Godot)

No nó Mimic (script MimicEnemy.gd):
- `vision_range`: Raio de visão (padrão: 20.0)
- `vision_angle`: Ângulo de visão em graus (padrão: 90.0)
- `patrol_speed`: Velocidade de patrulha (padrão: 2.0)
- `chase_speed_speed`: Velocidade modo velocidade (padrão: 4.0)
- `chase_strength_speed`: Velocidade modo força (padrão: 1.5)
- `speed_mode_damage`: Dano modo velocidade (padrão: 5.0)
- `strength_mode_damage`: Dano modo força (padrão: 25.0)
- `max_stamina`: Stamina máxima (padrão: 100.0)
- `stamina_drain_speed`: Drenagem stamina modo velocidade (padrão: 10.0)
- `stamina_drain_strength`: Drenagem stamina modo força (padrão: 20.0)
- `stamina_regen`: Regeneração stamina (padrão: 5.0)
- `attack_cooldown`: Cooldown de ataque (padrão: 2.0)
- `mode_switch_cooldown`: Cooldown de troca de modo (padrão: 5.0)

## Como Usar

1. Abra a cena `scenes/MimicExperiment.tscn` no Godot
2. Pressione F6 para rodar a cena
3. Use WASD para mover, Espaço para pular
4. O Mimic irá patrulhar até detectar o jogador
5. Quando detectado, entrará em modo de perseguição
6. O modo de perseguição muda automaticamente baseado na stamina

## Notas

- A NavigationMesh precisa ser configurada no editor para o pathfinding funcionar corretamente
- O modelo Mimic usa os materiais configurados em `Models/Mimic/DefMimic/materials/`
- As animações precisam estar disponíveis no AnimationPlayer do modelo Mimic
