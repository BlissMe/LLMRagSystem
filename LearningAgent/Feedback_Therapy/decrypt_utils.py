import binascii
import hashlib
import key_param
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def _get_key() -> bytes:
    """
    Derive a 32-byte AES key from the ENCRYPTION_KEY passphrase
    using SHA-256, to match typical AES-256-CBC setups.
    """
    if not key_param.ENCRYPTION_KEY:
        raise RuntimeError("ENCRYPTION_KEY is missing!")

    # Treat ENCRYPTION_KEY as a passphrase, not raw key
    secret_bytes = key_param.ENCRYPTION_KEY.encode("utf-8")
    return hashlib.sha256(secret_bytes).digest()  # 32 bytes -> 256-bit key



def decrypt_summary(enc: str) -> str:
    """
    Decrypt AES encrypted text in form:
    IV_HEX : CIPHERTEXT_HEX
    """
    if ":" not in enc:
        return ""

    iv_hex, cipher_hex = enc.split(":", 1)

    try:
        iv = binascii.unhexlify(iv_hex)
        ciphertext = binascii.unhexlify(cipher_hex)
    except Exception:
        return ""

    key = _get_key()

    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )

    decryptor = cipher.decryptor()
    padded_plain = decryptor.update(ciphertext) + decryptor.finalize()

    # remove PKCS7 padding
    pad_value = padded_plain[-1]
    if pad_value < 1 or pad_value > 16:
        return padded_plain.decode("utf-8", errors="ignore")

    plain = padded_plain[:-pad_value]
    return plain.decode("utf-8", errors="ignore")
