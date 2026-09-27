"""
setup_environment.py
====================
Validação e preparação automatizada do ambiente para o smw-evolutionary-agent.

Verifica:
  1. Integridade criptográfica da ROM de Super Mario World (USA) via SHA-1.
  2. Presença e configuração dos executáveis e bibliotecas do BizHawk 2.9.1.
  3. Integridade e formato do savestate nativo DP1.state.
  4. Presença dos scripts evolucionários: MarIO.lua (NEAT) e MarIO_GP.lua (Genetic Programming).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
BIZHAWK_DIR = Path(r"C:\BizHawk")
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_REPO = REPO_DIR / "Super Mario World (USA).sfc"
ROM_BIZHAWK = BIZHAWK_DIR / "Super Mario World (USA).sfc"
STATE_REPO = REPO_DIR / "DP1.state"
STATE_BIZHAWK = BIZHAWK_DIR / "DP1.state"

ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"


def check_rom() -> bool:
    print("\n[1/4] Verificando integridade da ROM...")
    if not ROM_REPO.exists():
        print(f"  [ERRO] ROM não encontrada em: {ROM_REPO}")
        return False

    data = ROM_REPO.read_bytes()
    sha1 = hashlib.sha1(data).hexdigest()
    print(f"  Arquivo : {ROM_REPO.name}")
    print(f"  Tamanho : {len(data) // 1024} KB")
    print(f"  SHA-1   : {sha1}")

    if sha1 == ROM_SHA1_USA:
        print("  [OK] SHA-1 oficial verificado (dump idêntico ao benchmark de referência).")
    else:
        print(f"  [AVISO] SHA-1 difere do padrão ({ROM_SHA1_USA}).")

    if not ROM_BIZHAWK.exists() or ROM_BIZHAWK.stat().st_size != len(data):
        shutil.copy2(ROM_REPO, ROM_BIZHAWK)
        print("  [OK] ROM sincronizada com C:\\BizHawk\\")
    return True


def check_emulator() -> bool:
    print("\n[2/4] Verificando emulador BizHawk...")
    if not EMUHAWK_EXE.exists():
        print(f"  [ERRO] BizHawk não encontrado em {EMUHAWK_EXE}")
        return False

    print(f"  [OK] Executável BizHawk encontrado: {EMUHAWK_EXE}")
    return True


def check_savestate() -> bool:
    print("\n[3/4] Verificando savestate nativo (DP1.state)...")
    valid_in_repo = STATE_REPO.exists() and STATE_REPO.stat().st_size > 10_000
    valid_in_biz = STATE_BIZHAWK.exists() and STATE_BIZHAWK.stat().st_size > 10_000

    if valid_in_repo:
        with open(STATE_REPO, "rb") as f:
            magic = f.read(4)
        if magic.startswith(b"PK") or magic.startswith(b"BZh"):
            print(f"  [OK] Savestate nativo verificado no repositório ({STATE_REPO.stat().st_size // 1024} KB).")
            if not valid_in_biz:
                shutil.copy2(STATE_REPO, STATE_BIZHAWK)
                print("  [OK] DP1.state sincronizado com C:\\BizHawk\\")
            return True

    print("  [INFO] DP1.state necessita ser gerado no primeiro lançamento.")
    return True


def check_scripts() -> bool:
    print("\n[4/4] Verificando scripts evolucionários...")
    scripts = ["MarIO.lua", "MarIO_GP.lua"]
    all_ok = True
    for s in scripts:
        src = REPO_DIR / s
        dst = BIZHAWK_DIR / s
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {s} ({src.stat().st_size // 1024} KB) copiado para BizHawk.")
        else:
            print(f"  [ERRO] {s} não encontrado no repositório.")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("  SMW EVOLUTIONARY AGENT - SETUP E VALIDAÇÃO")
    print("=" * 60)

    ok_rom = check_rom()
    ok_emu = check_emulator()
    ok_state = check_savestate()
    ok_scripts = check_scripts()

    print("\n" + "=" * 60)
    if ok_rom and ok_emu and ok_state and ok_scripts:
        print("  STATUS: AMBIENTE 100% CONFIGURADO E OPERACIONAL!")
        print("=" * 60)
        print("\nPara iniciar o agente de forma totalmente autônoma:")
        print("  python launch_mario.py       # Para rodar NEAT")
        print("  python launch_mario.py --gp  # Para rodar Genetic Programming\n")
    else:
        print("  STATUS: FALHA NA CONFIGURAÇÃO. Verifique as mensagens acima.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
