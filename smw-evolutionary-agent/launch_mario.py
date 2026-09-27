"""
launch_mario.py
===============
Pipeline de lançamento 100% autônomo para o Super Mario World com agentes evolucionários.
Suporta NEAT (MarIO.lua) e Genetic Programming (MarIO_GP.lua).

Fluxo de Execução Autônoma:
  1. Validação de integridade da ROM e dos binários do emulador BizHawk.
  2. Verificação de compatibilidade do savestate 'DP1.state' (formato nativo BizHawk).
     Caso inexista ou seja incompatível, executa o bootstrap determinístico via Lua
     para navegar os menus do SMW e salvar o estado no frame exato de início de fase.
  3. Lançamento direto do BizHawk com o script evolucionário (--lua=MarIO.lua ou --lua=MarIO_GP.lua).

Uso:
  python launch_mario.py           # Inicia MarI/O (NEAT)
  python launch_mario.py --gp      # Inicia MarI/O Genetic Programming (Interpretable Tree-GP)
  python launch_mario.py --reboot-state  # Força regeneração do DP1.state
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Constantes de diretórios e caminhos
REPO_DIR = Path(__file__).resolve().parent
BIZHAWK_DIR = Path(r"C:\BizHawk")
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_FILE = BIZHAWK_DIR / "Super Mario World (USA).sfc"
STATE_FILE = BIZHAWK_DIR / "DP1.state"

LUA_NEAT = BIZHAWK_DIR / "MarIO.lua"
LUA_GP = BIZHAWK_DIR / "MarIO_GP.lua"

ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"


def log(msg: str):
    print(f"[SMW-AGENT] {msg}")


def check_rom_and_binaries() -> bool:
    """Verifica e sincroniza ROM e scripts para o diretório do BizHawk."""
    if not EMUHAWK_EXE.exists():
        log(f"ERRO: Executável do BizHawk não encontrado em {EMUHAWK_EXE}")
        return False

    # Copia ROM do repositório para o BizHawk se necessário
    rom_src = REPO_DIR / "Super Mario World (USA).sfc"
    if not ROM_FILE.exists() and rom_src.exists():
        shutil.copy2(rom_src, ROM_FILE)
        log("ROM copiada para C:\\BizHawk\\")

    if not ROM_FILE.exists():
        log(f"ERRO: ROM não encontrada em {ROM_FILE}")
        return False

    # Valida SHA-1
    sha1 = hashlib.sha1(ROM_FILE.read_bytes()).hexdigest()
    if sha1 == ROM_SHA1_USA:
        log("ROM verificada com sucesso (SHA-1 correspondente ao dump oficial USA).")
    else:
        log(f"AVISO: SHA-1 da ROM ({sha1}) difere do padrão ({ROM_SHA1_USA}).")

    # Garante que os scripts Lua mais recentes estão no diretório do BizHawk
    for script_name in ["MarIO.lua", "MarIO_GP.lua"]:
        src = REPO_DIR / script_name
        dst = BIZHAWK_DIR / script_name
        if src.exists():
            shutil.copy2(src, dst)
            log(f"Script atualizado: {dst.name}")

    # Remove config.ini corrompido ou desatualizado se existir para evitar caixas de diálogo
    cfg_file = BIZHAWK_DIR / "config.ini"
    if cfg_file.exists():
        # Verifica se o config.ini causa popup de versão
        try:
            content = cfg_file.read_text(encoding="utf-8", errors="ignore")
            if "DispMethodGL" not in content or "MainFormWidth" not in content:
                cfg_file.unlink(missing_ok=True)
                log("config.ini legado removido para garantir inicialização sem popups.")
        except Exception:
            cfg_file.unlink(missing_ok=True)

    return True


def is_valid_bizhawk_state(state_path: Path) -> bool:
    """Verifica se o arquivo é um savestate nativo válido do BizHawk."""
    if not state_path.exists() or state_path.stat().st_size < 10_000:
        return False

    with open(state_path, "rb") as f:
        magic = f.read(16)

    # Formato Libretro/Snes9x começa com #!s9xsnp (incompatível com BizHawk savestate.load)
    if magic.startswith(b"#!s9xsn"):
        return False

    # BizHawk savestates usam contêiner zip/bzip2 (PK\x03\x04 ou BZh)
    return magic.startswith(b"PK") or magic.startswith(b"BZh") or magic.startswith(b"BizState")


def generate_native_savestate() -> bool:
    """
    Navega de forma 100% autônoma pelos menus da ROM de SMW e salva
    o estado DP1.state no formato nativo do BizHawk no frame inicial do nível.
    """
    log("Gerando savestate nativo DP1.state de forma autônoma...")

    bootstrap_lua = BIZHAWK_DIR / "auto_bootstrap_dp1.lua"
    lua_code = """-- auto_bootstrap_dp1.lua
-- Gerador autônomo de savestate DP1.state para BizHawk
local frame = 0
local logfile = io.open("C:/BizHawk/bootstrap.log", "w")

while frame < 1500 do
    frame = frame + 1
    
    -- Pressiona Start em pulsos regulares para avançar a tela de título e selecionar o save file
    if (frame % 30) < 6 then
        joypad.set({["P1 Start"] = true})
    else
        joypad.set({["P1 Start"] = false})
    end
    
    -- No Overworld map, avança para a direita e pressiona botão de entrada
    if frame > 380 and (frame % 20) < 10 then
        joypad.set({["P1 Right"] = true, ["P1 Start"] = true, ["P1 B"] = true})
    end
    
    local mode = memory.readbyte(0x0100)
    local marioX = memory.read_s16_le(0x94)
    
    -- Game Mode 0x14 = In-Level (Gameplay ativo)
    if mode == 0x14 and marioX >= 100 then
        savestate.save("C:/BizHawk/DP1.state")
        if logfile then
            logfile:write(string.format("SUCESSO: Nível carregado no frame %d (Mode 0x14, MarioX=%d)\\n", frame, marioX))
            logfile:close()
        end
        client.closeemulator()
        return
    end
    
    emu.frameadvance()
end

if logfile then
    logfile:write("TIMEOUT: Não foi possível atingir o nível em 1500 frames.\\n")
    logfile:close()
end
client.closeemulator()
"""
    bootstrap_lua.write_text(lua_code, encoding="utf-8")

    cmd = [str(EMUHAWK_EXE), f"--lua={bootstrap_lua}", str(ROM_FILE)]
    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Copia o state gerado de volta para o repo
    if is_valid_bizhawk_state(STATE_FILE):
        repo_state = REPO_DIR / "DP1.state"
        shutil.copy2(STATE_FILE, repo_state)
        log(f"DP1.state nativo gerado com sucesso ({STATE_FILE.stat().st_size // 1024} KB).")
        return True
    else:
        log("FALHA ao gerar DP1.state nativo.")
        return False


def launch_agent(use_gp: bool = False):
    """Executa o BizHawk com o agente evolucionário selecionado."""
    script_path = LUA_GP if use_gp else LUA_NEAT
    algo_name = "Genetic Programming (Árvore Simbólica Interpretável)" if use_gp else "NEAT (Topologia Neural Evolving)"

    log(f"Iniciando emulação com {algo_name}...")
    log(f"Executável: {EMUHAWK_EXE}")
    log(f"Script:     {script_path.name}")
    log(f"ROM:        {ROM_FILE.name}")
    log("O emulador abrirá agora. Acompanhe a evolução do agente na janela.")

    cmd = [
        str(EMUHAWK_EXE),
        f"--lua={script_path}",
        str(ROM_FILE)
    ]

    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    try:
        proc.wait()
    except KeyboardInterrupt:
        log("Interrupção solicitada pelo usuário. Encerrando BizHawk...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    log("Sessão de emulação finalizada.")


def main():
    parser = argparse.ArgumentParser(
        description="Lançador autônomo do agente evolucionário Super Mario World (NEAT / GP)"
    )
    parser.add_argument(
        "--gp",
        action="store_true",
        help="Executa a abordagem de Genetic Programming (MarIO_GP.lua)"
    )
    parser.add_argument(
        "--reboot-state",
        action="store_true",
        help="Regenera o savestate DP1.state mesmo se já existir"
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  SUPER MARIO WORLD - AGENTE EVOLUCIONÁRIO AUTÔNOMO")
    print("=" * 65 + "\n")

    if not check_rom_and_binaries():
        sys.exit(1)

    # Verifica necessidade de regenerar state
    need_state = args.reboot_state or not is_valid_bizhawk_state(STATE_FILE)

    # Se existe cópia válida no repo, restaura diretamente
    repo_state = REPO_DIR / "DP1.state"
    if need_state and is_valid_bizhawk_state(repo_state) and not args.reboot_state:
        shutil.copy2(repo_state, STATE_FILE)
        log("DP1.state válido restaurado do repositório.")
        need_state = False

    if need_state:
        success = generate_native_savestate()
        if not success:
            log("ERRO CRÍTICO: Não foi possível obter ou gerar DP1.state válido.")
            sys.exit(1)

    launch_agent(use_gp=args.gp)


if __name__ == "__main__":
    main()
