#!/usr/bin/env python3
"""
Self-extracting launcher for the trader MVP.
Run from the project root:
    python -m src.agents.macro_agent
"""

import base64
import datetime as dt
import os
import shutil
import tempfile
import textwrap
import zipfile

ZIP_B64 = """
UEsDBBQAAAAIAJJQaVg7GJ2gjwEAAKMFAAARAAAAdHJhZGVyL1JFQURNRS5tZHWQwU7DMAyG
7/sVnLOkSVEaWbp1wg5cgB2YxLrNpmEpJKm7cf++TlLXXZfpZP2S9fP6yYk+oA6
""".strip()  # <-- still truncated for brevity; paste the full string here

def main() -> None:
    dest = "trader-mvp"
    if os.path.exists(dest):
        shutil.rmtree(dest)

    # --- clean & pad the base-64 payload ---
    b64_clean = ZIP_B64.replace("\n", "").replace(" ", "")
    b64_clean += "=" * (-len(b64_clean) % 4)

    data = base64.b64decode(b64_clean)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(data)
        tmp.flush()
        with zipfile.ZipFile(tmp.name, "r") as zf:
            zf.extractall(dest)

    print(f"✅ Project extracted to ./{dest}")
    print("Next:")
    print(textwrap.dedent(f"""
        cd {dest}
        python -m pip install -r requirements.txt
        python -m src.main --help
    """))

if __name__ == "__main__":
    main()