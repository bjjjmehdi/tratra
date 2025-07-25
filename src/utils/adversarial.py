"""PNG validation & adversarial defense."""
import hashlib
from typing import Final

# reject known adversarial hashes
_BLACKLIST: Final = {
    bytes.fromhex("deadbeef"),  # placeholder
}

def validate_png(data: bytes) -> None:
    """Raise if PNG is adversarial or malformed."""
    if len(data) < 100 or len(data) > 5 * 1024 * 1024:
        raise ValueError("PNG size out of bounds")
    h = hashlib.sha256(data).digest()
    if h in _BLACKLIST:
        raise ValueError("Adversarial PNG detected")
    # minimal PNG header check
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG")