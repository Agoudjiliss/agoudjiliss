"""
Auth Picture – Cryptography Service
Self-signed certificate generation for C2PA signing.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

logger = logging.getLogger("auth_picture.crypto")


def ensure_certs(certs_dir: Path) -> tuple[Path, Path]:
    """Ensure self-signed cert and key exist. Creates them if missing.

    Returns:
        (cert_path, key_path)
    """
    cert_path = certs_dir / "auth_picture.pem"
    key_path = certs_dir / "auth_picture.key"

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    logger.info("Generating self-signed certificate for C2PA...")
    certs_dir.mkdir(parents=True, exist_ok=True)

    # Generate EC key (P-256)
    private_key = ec.generate_private_key(ec.SECP256R1())

    # Build self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "AuthPicture Self-Signed"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AuthPicture"),
    ])

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .sign(private_key, hashes.SHA256())
    )

    # Write PEM files
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))

    logger.info("Certificate saved to %s", cert_path)
    return cert_path, key_path
