# Semantic Twin Engine - Installation

## Installation rapide

```bash
cd /Users/joachimbischpeuchet/src/semantic-twin/semantic-twin-code

# Installation avec pip (recommandé)
pip install -e .

# OU avec requirements.txt (legacy)
pip install -r semantic_twin_engine/requirements.txt
```

## Installation pour développement

```bash
# Avec dépendances de développement
pip install -e ".[dev]"

# Cela installe aussi : pytest, black, ruff, mypy
```

## Configuration

Copiez et configurez le fichier `.env` :

```bash
cp semantic_twin_engine/.env.example semantic_twin_engine/.env
```

Éditez `.env` pour choisir votre provider AI (OpenAI ou Mistral).

## Voir aussi

- [AI_PROVIDERS.md](AI_PROVIDERS.md) - Guide des providers AI
- [GUIDE_MISTRAL.md](GUIDE_MISTRAL.md) - Guide rapide Mistral
- [DEPENDENCY_MANAGEMENT.md](../DEPENDENCY_MANAGEMENT.md) - Gestion des dépendances
