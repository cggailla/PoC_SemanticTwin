# 🚀 Guide rapide : Passer à Mistral

## Pour utiliser Mistral Large au lieu d'OpenAI

### Étape 1 : Obtenir une clé API Mistral

1. Créez un compte sur https://console.mistral.ai
2. Allez dans la section "API Keys"
3. Créez une nouvelle clé et copiez-la

### Étape 2 : Installer le SDK Mistral

```bash
cd /Users/joachimbischpeuchet/src/semantic-twin/semantic-twin-code/semantic_twin_engine
pip install mistralai
```

### Étape 3 : Configurer le fichier .env

Éditez le fichier `.env` :

```bash
# Changer le provider
AI_PROVIDER=mistral

# Décommenter et ajouter votre clé Mistral
MISTRAL_API_KEY=votre-clé-api-mistral-ici
MISTRAL_MODEL=mistral-large-latest
MISTRAL_EMBEDDING_MODEL=mistral-embed

# Désactiver le mode fake pour utiliser l'API réelle
FAKE_OPENAI_RESULT=false
```

### Étape 4 : Tester la configuration

```bash
python test_provider_config.py
```

Vous devriez voir :

```
============================================================
🧪 TEST DE CONFIGURATION AI PROVIDER
============================================================

📡 Provider configuré: MISTRAL
   └─ Modèle: mistral-large-latest
   └─ Embeddings: mistral-embed
   └─ Dimensions: 1024
   └─ API Key: ✅ Configurée
   └─ SDK Mistral: ✅ Installé (v0.x.x)

🎭 Mode Fake (test sans API): ❌ Désactivé

============================================================
✅ Configuration valide!
============================================================
```

### Étape 5 : Lancer le serveur Flask

```bash
python app.py
```

Le serveur utilisera automatiquement Mistral !

## Pour revenir à OpenAI

Simplement changer dans `.env` :

```bash
AI_PROVIDER=openai
```

## Comparaison des coûts

| Provider    | Embeddings (1M tokens) | Texte (1M tokens input) |
| ----------- | ---------------------- | ----------------------- |
| **OpenAI**  | $0.02                  | $2.50 (GPT-4o)          |
| **Mistral** | $0.10                  | $2.00 (Large)           |

Mistral est **plus économique** pour le texte, mais légèrement plus cher pour les embeddings.

## Avantages de Mistral

✅ Prix compétitifs  
✅ Conformité RGPD (hébergé en Europe)  
✅ Performance comparable à GPT-4  
✅ Open source friendly

## Questions fréquentes

**Q: Puis-je utiliser les deux en même temps ?**  
R: Non, vous devez choisir un provider à la fois via `AI_PROVIDER`.

**Q: Mes anciens rapports fonctionneront-ils ?**  
R: Les rapports existants sont indépendants du provider. Seule la génération de nouveaux audits utilise le provider configuré.

**Q: Les embeddings sont-ils compatibles ?**  
R: Non, OpenAI (1536 dims) et Mistral (1024 dims) ont des dimensions différentes. Le cache est automatiquement invalidé si vous changez de provider.
