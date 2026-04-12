# ML Survival Arena (Godot + ONNX)

Projeto de jogo 3D em Godot onde um agente humanoide precisa sobreviver o maximo de tempo possivel de um perseguidor em formato de capsula rigida.

## O que foi implementado

- Agente humanoide com gravidade e controle por acoes continuas.
- Cada parte principal do corpo move separadamente (cabeca, bracos, pernas).
- Perseguidor fisico com RigidBody3D em formato de capsula.
- Loop de episodios com reset automatico, limite de tempo e condicao de captura.
- Interface rica com informacoes de:
  - jogo (tempo atual, melhor tempo, distancia, estado)
  - agente/perseguidor (velocidades, altura, acoes)
  - treino (geracao, fitness, checkpoint, estado do ONNX)
- Pipeline de treino em Python com salvamento de checkpoint e exportacao ONNX.
- Reuso de checkpoint entre treinos para acelerar convergencia (opcao na interface).
- Servidor local de inferencia ONNX via UDP para o Godot consultar a politica.

## Estrutura principal

- Cena principal: res://scenes/Main.tscn
- Agente: res://scenes/HumanoidAgent.tscn
- Perseguidor: res://scenes/Pursuer.tscn
- HUD: res://scenes/HUD.tscn
- Treino: res://ml/train_agent.py
- Inferencia ONNX: res://ml/onnx_inference_server.py

## Requisitos

- Godot 4.5+
- Python 3.10+ (testado com 3.14)

Instale dependencias Python:

```bash
pip install -r ml/requirements.txt
```

## Como executar

1. Abra o projeto no Godot.
2. Rode a cena principal (ja configurada como main scene).
3. Se existir ONNX em ml/models/agent_policy.onnx e o toggle ONNX estiver ativo, o jogo tenta iniciar o servidor de inferencia automaticamente.
4. Se nao existir ONNX, o agente usa politica heuristica ate voce treinar.

## Como treinar

Pela interface dentro do jogo:

1. Ajuste Geracoes, Populacao, Tempo de episodio e Seed.
2. Marque ou desmarque Reusar checkpoint no treino.
3. Clique em Iniciar Treino.
4. Acompanhe metricas no painel Treino.

Ao concluir, os arquivos principais gerados sao:

- ml/models/agent_policy.pt (checkpoint para continuar treino)
- ml/models/agent_policy.onnx (modelo para inferencia)
- ml/models/training_status.json (status em tempo real)
- ml/models/training_history.jsonl (historico por geracao)

## Treino manual via terminal

```bash
python ml/train_agent.py --model-dir ml/models --generations 60 --population 28 --episode-seconds 45 --reuse true
```

## Inferencia manual via terminal

```bash
python ml/onnx_inference_server.py --model ml/models/agent_policy.onnx --host 127.0.0.1 --port 8765
```

## Observacoes

- O modelo FBX em Models/RobotKyle.fbx e carregado no agente quando disponivel.
- O rig procedural (capsulas/esferas) continua ativo para garantir o movimento separado das partes do corpo.
- Se o Python nao estiver no PATH, ajuste python_command nos scripts OnnxPolicyClient.gd e TrainingManager.gd.
