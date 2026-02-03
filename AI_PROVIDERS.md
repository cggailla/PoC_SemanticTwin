# Configuration des Providers AI (OpenAI ou Mistral)

## Vue d'ensemble

Le Semantic Twin Engine supporte maintenant deux providers AI :

- **OpenAI** (GPT-4, text-embedding-3-small)
- **Mistral AI** (mistral-large-latest, mistral-embed)

Vous pouvez choisir le provider via la variable d'environnement `AI_PROVIDER`.

## Configuration

### 1. Fichier `.env`

Copiez le fichier d'exemple :

```bash
cp semantic_twin_engine/.env.example semantic_twin_engine/.env
```

### 2. Choisir le provider

Éditez `semantic_twin_engine/.env` et définissez :

#### Option A : Utiliser OpenAI

```bash
AI_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### Option B : Utiliser Mistral

```bash
AI_PROVIDER=mistral
MISTRAL_API_KEY=votre-clé-mistral
MISTRAL_MODEL=mistral-large-latest
MISTRAL_EMBEDDING_MODEL=mistral-embed
```

## Différences techniques

| Caractéristique             | OpenAI                 | Mistral                              |
| --------------------------- | ---------------------- | ------------------------------------ |
| **Modèle de texte**         | gpt-4o, gpt-4-turbo    | mistral-large-latest, mistral-medium |
| **Modèle d'embeddings**     | text-embedding-3-small | mistral-embed                        |
| **Dimensions d'embeddings** | 1536                   | 1024                                 |
| **SDK Python**              | `openai`               | `mistralai`                          |

## Installation des dépendances

Les deux SDKs sont inclus dans `requirements.txt` :

```bash
pip install -r semantic_twin_engine/requirements.txt
```

## Obtenir les clés API

### OpenAI

1. Créez un compte sur https://platform.openai.com
2. Allez dans API Keys
3. Créez une nouvelle clé API
4. Copiez-la dans `OPENAI_API_KEY`

### Mistral AI

1. Créez un compte sur https://console.mistral.ai
2. Allez dans API Keys
3. Créez une nouvelle clé API
4. Copiez-la dans `MISTRAL_API_KEY`

## Tests sans API (Mode Fake)

Pour tester sans consommer d'API :

```bash
FAKE_OPENAI_RESULT=true
```

Cela génère des embeddings déterministes sans appeler d'API.

## Changement de provider

Pour changer de provider, modifiez simplement `AI_PROVIDER` dans votre `.env` :

```bash
# Passer d'OpenAI à Mistral
AI_PROVIDER=mistral

# Passer de Mistral à OpenAI
AI_PROVIDER=openai
```

⚠️ **Note** : Le cache d'embeddings est invalide si vous changez de modèle, car les dimensions sont différentes (1536 vs 1024).

## Compatibilité avec l'API

L'API Flask (`/api/audit`) accepte désormais des paramètres optionnels :

```json
{
  "entity_name": "Ma Société",
  "legacy_keywords": ["ancien", "traditionnel"],
  "strategy_keywords": ["innovation", "digital"],
  "model": "mistral-large-latest", // optionnel
  "embedding_model": "mistral-embed" // optionnel
}
```

Si non spécifié, le provider configuré dans `.env` est utilisé.
