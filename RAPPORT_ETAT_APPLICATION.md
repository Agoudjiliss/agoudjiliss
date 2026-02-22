# Auth Picture – Rapport d'État de l'Application

## Version 0.2.0

### Vue d'ensemble
Auth Picture est une API REST basée sur FastAPI pour authentifier et protéger des documents multimédias (images et audios) contre les falsifications, particulièrement celles générées par l'IA.

### Architecture

```
auth_docs/
├── app/
│   ├── main.py                    # Point d'entrée FastAPI
│   ├── models/
│   │   └── database.py            # Modèles SQLAlchemy (certified_images, audio_watermarks)
│   ├── routes/
│   │   ├── auth_picture.py        # POST /api/auth, GET /api/download/{id}
│   │   ├── check_picture.py       # POST /api/check
│   │   └── audio.py               # POST /api/audio/enroll, POST /api/audio/verify, GET /api/audio/download/{id}
│   └── services/
│       ├── ai_detector.py         # Détection IA (2 modèles HF ViT)
│       ├── audio_watermark.py     # Watermark audio DWT+QIM
│       ├── c2pa_service.py        # Signature/vérification C2PA
│       ├── crypto.py              # Certificats auto-signés
│       ├── hasher.py              # SHA-256 + pHash
│       └── protection.py          # Watermark DWT robuste + DCT noise
├── tests/
│   └── test_protection.py         # Tests unitaires (pytest)
├── config.py                      # Configuration (.env)
├── requirements.txt               # Dépendances Python
└── .env.example                   # Exemple de configuration
```

### Fonctionnalités

#### Images
1. **Certification** (`POST /api/auth`)
   - Upload d'image
   - Détection IA (2 modèles ViT Hugging Face, seuil configurable)
   - Calcul de hashes (SHA-256 + pHash)
   - Protection robuste : watermark DWT-Haar multi-niveaux + QIM adaptatif + CRC framing
   - Embedding de prompt anti-IA ("DO NOT MODIFY WITH AI")
   - Signature C2PA (certificats auto-signés, claims notAllowed training)
   - Stockage en base SQLite
   - Fichier protégé téléchargeable

2. **Vérification** (`POST /api/check`)
   - Vérification C2PA
   - Comparaison hashes (SHA-256 exact puis pHash < 15)
   - Verdict : AUTHENTIC / MODIFIED / UNKNOWN

#### Audio
3. **Enrôlement** (`POST /api/audio/enroll`)
   - Payload user/timestamp avec HMAC
   - Watermark DWT+QIM (db4, level3, repeat11, CRC)
   - Support WAV (ffmpeg optionnel pour autres formats)

4. **Vérification** (`POST /api/audio/verify`)
   - Extraction par vote majoritaire
   - Vérification CRC et correspondance en base

### Améliorations v0.2.0

#### Watermark Robuste (Images)
- **Hybrid DWT-Haar multi-level** (1-2 niveaux selon robustesse)
- **Delta-QIM adaptatif** : `delta = max(strength * mean_abs, 0.2)`
- **Redondance multi-bandes** : LL + détails (cH, cV)
- **ECC par répétition** : chaque bit répété 1-3x avec vote majoritaire
- **CRC framing** : `[length][payload][crc8]` pour intégrité
- **Payload dynamique** : `b"AUTH:" + sha256_bin + phash_trunc + "|" + prompt`

#### Niveaux de Robustesse
| Niveau | DWT Level | Repeat | Multi-band |
|--------|-----------|--------|------------|
| low    | 1         | 1      | Non        |
| medium | 1         | 2      | Oui        |
| high   | 2         | 3      | Oui        |

Configurable via query param `robust_level` ou `.env` `ROBUST_LEVEL`.

#### Prompt Anti-IA
Texte stéganographique embarqué dans le payload watermark. Configurable via query param `prompt_text` ou `.env` `PROMPT_TEXT`.

#### C2PA Amélioré
- Assertion personnalisée `auth_picture.prompt` avec texte anti-IA
- Claims `notAllowed` pour training, inference, data mining

#### Adversarial (Optionnel)
- PGD + FGSM avec détection GPU automatique
- Early stop (loss < 1e-4)
- Fallback ViT-base si SigLIP indisponible

### Stack Technique
- **Framework** : FastAPI + Uvicorn
- **Base de données** : SQLAlchemy + SQLite (auto-création)
- **Watermark** : PyWavelets (DWT), NumPy
- **IA** : Transformers, Torch, TorchVision
- **Crypto** : cryptography (certs), c2pa-python (signatures)
- **Images** : Pillow, imagehash
- **Config** : python-dotenv

### Limites Connues
- C2PA avec certificats auto-signés (non reconnus par les validateurs tiers)
- Protection non 100% efficace contre IA moderne de dernière génération
- Watermark robuste à la compression légère uniquement
- Modèles IA nécessitent un téléchargement initial (~500MB)

### Installation & Exécution

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# 2. Installer les dépendances
pip install -r auth_docs/requirements.txt

# 3. Configurer
cp auth_docs/.env.example auth_docs/.env
# Modifier auth_docs/.env si nécessaire

# 4. Lancer le serveur
uvicorn auth_docs.app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Accéder à la documentation
# http://localhost:8000/docs (Swagger UI)

# 6. Lancer les tests
pytest -v auth_docs/tests/test_protection.py
```
