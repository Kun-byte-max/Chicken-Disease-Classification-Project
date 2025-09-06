from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
print("[DEBUG] ROOT_DIR is:", ROOT_DIR)   # 👈 add this

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"

print("[DEBUG] CONFIG_FILE_PATH is:", CONFIG_FILE_PATH)
print("[DEBUG] PARAMS_FILE_PATH is:", PARAMS_FILE_PATH)
