#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration du provider AI.
"""

import os
import sys
from pathlib import Path

# Ajouter le chemin du package
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def test_provider_config():
    """Test la configuration du provider."""
    provider = os.getenv("AI_PROVIDER", "openai").lower()

    print("=" * 60)
    print("🧪 TEST DE CONFIGURATION AI PROVIDER")
    print("=" * 60)
    print(f"\n📡 Provider configuré: {provider.upper()}")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = 1536

        print(f"   └─ Modèle: {model}")
        print(f"   └─ Embeddings: {embed_model}")
        print(f"   └─ Dimensions: {dimensions}")
        print(f"   └─ API Key: {'✅ Configurée' if api_key else '❌ Manquante'}")

        if not api_key:
            print("\n⚠️  OPENAI_API_KEY n'est pas définie dans .env")
            return False

    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        embed_model = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
        dimensions = 1024

        print(f"   └─ Modèle: {model}")
        print(f"   └─ Embeddings: {embed_model}")
        print(f"   └─ Dimensions: {dimensions}")
        print(f"   └─ API Key: {'✅ Configurée' if api_key else '❌ Manquante'}")

        if not api_key:
            print("\n⚠️  MISTRAL_API_KEY n'est pas définie dans .env")
            return False

        # Vérifier si le SDK est installé
        try:
            import mistralai

            print(f"   └─ SDK Mistral: ✅ Installé (v{mistralai.__version__})")
        except ImportError:
            print("   └─ SDK Mistral: ❌ Non installé")
            print("\n⚠️  Installer avec: pip install mistralai")
            return False
    else:
        print(f"\n❌ Provider inconnu: {provider}")
        print("   Les valeurs valides sont: 'openai' ou 'mistral'")
        return False

    # Test mode fake
    fake_mode = os.getenv("FAKE_OPENAI_RESULT", "false").lower() in ("true", "1", "yes")
    print(
        f"\n🎭 Mode Fake (test sans API): {'✅ Activé' if fake_mode else '❌ Désactivé'}"
    )

    print("\n" + "=" * 60)
    print("✅ Configuration valide!")
    print("=" * 60)

    return True


def test_imports():
    """Test que les modules s'importent correctement."""
    print("\n🔍 Test des imports...")

    try:
        from core.embedding_store import FakeEmbeddingProvider

        print("   ✅ EmbeddingStore importé")
        print("   ✅ FakeEmbeddingProvider importé")

        # Test dimension detection
        openai_dim = FakeEmbeddingProvider.get_dimension_for_model(
            "text-embedding-3-small"
        )
        mistral_dim = FakeEmbeddingProvider.get_dimension_for_model("mistral-embed")

        print(f"   ✅ Dimensions OpenAI: {openai_dim}")
        print(f"   ✅ Dimensions Mistral: {mistral_dim}")

        assert openai_dim == 1536, "OpenAI dimensions should be 1536"
        assert mistral_dim == 1024, "Mistral dimensions should be 1024"

        return True
    except Exception as e:
        print(f"   ❌ Erreur d'import: {e}")
        return False


if __name__ == "__main__":
    print("\n")
    config_ok = test_provider_config()

    if config_ok:
        imports_ok = test_imports()
        if imports_ok:
            print("\n🎉 Tous les tests sont passés!\n")
            sys.exit(0)

    print("\n❌ Certains tests ont échoué.\n")
    sys.exit(1)
