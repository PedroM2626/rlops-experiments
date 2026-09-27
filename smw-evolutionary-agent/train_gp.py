"""
train_gp.py
===========
Gerenciador de treinamento autônomo do agente Genetic Programming (MarIO_GP).

Executa o BizHawk em modo turbo, monitora o log de evolução gerado pelas
gerações do GP, persiste os checkpoints e extrai a política explícita final.
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
LUA_SCRIPT = BIZHAWK_DIR / "MarIO_GP.lua"
LOG_FILE = BIZHAWK_DIR / "gp_training_results.txt"


def run_training(duration_seconds: int = 180):
    print("=" * 65)
    print("  INICIANDO SESSÃO DE TREINAMENTO REAL DO AGENTE GP")
    print("=" * 65)
    print(f"Duração planejada: ~{duration_seconds} segundos (modo turbo)")
    print(f"Emulador: {EMUHAWK_EXE}")
    print(f"Script:   {LUA_SCRIPT}")
    print(f"ROM:      {ROM_FILE.name}\n")

    # Limpa log anterior se existir
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    cmd = [str(EMUHAWK_EXE), f"--lua={LUA_SCRIPT}", str(ROM_FILE)]
    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    start_time = time.time()
    last_printed_lines = 0

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(3)
            elapsed = int(time.time() - start_time)
            if LOG_FILE.exists():
                try:
                    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if len(lines) > last_printed_lines:
                        for l in lines[last_printed_lines:]:
                            if l.startswith("Gen ") or l.startswith("--- MELHOR POLITICA"):
                                print(f"[{elapsed:3d}s] {l}")
                        last_printed_lines = len(lines)
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\nInterrupção do usuário recebida.")
    finally:
        print(f"\nFinalizando sessão de emulação ({int(time.time() - start_time)}s decorridos)...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Copia checkpoints e logs para o repositório
    if LOG_FILE.exists():
        shutil.copy2(LOG_FILE, REPO_DIR / "gp_training_results.txt")
        print("\n[OK] Resultados gravados em gp_training_results.txt")

    # Copia .gppool gerados
    pool_files = list(BIZHAWK_DIR.glob("*.gppool"))
    for f in pool_files:
        shutil.copy2(f, REPO_DIR / f.name)
        print(f"[OK] Checkpoint preservado: {f.name}")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    run_training(dur)
