"""
train_neat.py
=============
Gerenciador de benchmark pareado do agente NEAT (MarIO original).

Executa o BizHawk em modo turbo (speedmode 600) por tempo idêntico ao GP (180s),
monitora a evolução das espécies e genomas, persiste os checkpoints e extrai
as métricas de fitness para comparação direta.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BIZHAWK_DIR = Path(r"C:\BizHawk")
REPO_DIR = Path(__file__).resolve().parent
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_FILE = BIZHAWK_DIR / "Super Mario World (USA).sfc"
LUA_SCRIPT = BIZHAWK_DIR / "MarIO.lua"
LOG_FILE = BIZHAWK_DIR / "neat_training_results.txt"


def run_neat_benchmark(duration_seconds: int = 180):
    print("=" * 65)
    print("  INICIANDO SESSÃO DE BENCHMARK DO AGENTE NEAT (PAREADO)")
    print("=" * 65)
    print(f"Duração planejada: {duration_seconds} segundos (speedmode 600)")
    print(f"Emulador: {EMUHAWK_EXE}")
    print(f"Script:   {LUA_SCRIPT}")
    print(f"ROM:      {ROM_FILE.name}\n")

    if LOG_FILE.exists():
        LOG_FILE.unlink()

    cmd = [str(EMUHAWK_EXE), f"--lua={LUA_SCRIPT}", str(ROM_FILE)]
    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    start_time = time.time()
    last_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            if LOG_FILE.exists():
                try:
                    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if len(lines) > last_count:
                        last_count = len(lines)
                        print(f"[{elapsed:3d}s] Avaliações: {len(lines)} | Último: {lines[-1]}")
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\nInterrupção recebida.")
    finally:
        print(f"\nFinalizando sessão NEAT ({int(time.time() - start_time)}s decorridos)...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Copia resultados para o repositório
    if LOG_FILE.exists():
        shutil.copy2(LOG_FILE, REPO_DIR / "neat_training_results.txt")
        print("\n[OK] Resultados gravados em neat_training_results.txt")

    for f in BIZHAWK_DIR.glob("*.pool"):
        shutil.copy2(f, REPO_DIR / f.name)
        print(f"[OK] Pool preservado: {f.name}")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    run_neat_benchmark(dur)
