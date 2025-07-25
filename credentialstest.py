# credentialstest.py  ← 放在 project/ 根目录
from utils.config import load_config

cfg = load_config()
print("kimi_key =", cfg["model"]["kimi_key"])