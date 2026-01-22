# Semantic Twin Engine (PoC)

**Architect a deterministic, audit-grade probing engine to map corporate entities within LLM latent spaces.**

This Proof of Concept (PoC) is a SaaS Backend engine designed to probe Large Language Models (LLMs) to extract vector embeddings and logits, enabling the creation of a "Semantic Twin" of an organization. It prioritizes reproducibility (`temperature=0`), statelessness, and strict type safety.

## 🚀 Features

- **Deterministic Probing**: Enforces strict parameters for reproducible results.
- **Modular Architecture**: Core engine separated from pluggable probe modules.
- **Vector Analysis**: Extraction and comparison of embeddings using Cosine Similarity.
- **Logit Analysis**: Analysis of model confidence and token probabilities.
- **Visual Reporting**: Generates interactive HTML dashboards (Plotly) for audit results.
- **Resilience**: Robust error handling and API retry mechanisms.

## 📋 Prerequisites

- **Python 3.10+**
- An **OpenAI API Key**

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd PoC_SemanticTwin
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r semantic_twin_engine/requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the `semantic_twin_engine` directory (or root if supported by config loader) based on your needs.
    ```bash
    # semantic_twin_engine/.env
    OPENAI_API_KEY=sk-...
    ```

## 🏃 Usage

The engine is designed to be run as a module or script. The main entry point executes the registered probes defined in your configuration.

**Run the Engine:**

```bash
python semantic_twin_engine/main.py
```

**Output:**

- **Logs**: Execution details are logged to the console/files.
- **Audit Report**: A JSON report is generated in `output/` (or configured path).
- **Visual Dashboard**: An HTML report is generated in `output/visuals/` containing interactive plots of the semantic analysis.

## 📂 Project Structure

- **`semantic_twin_engine/`**: Main package.
  - **`core/`**: The immutable engine logic (Orchestrator, I/O, Config).
  - **`modules/`**: Pluggable logic (Probes, Analysis Tools).
  - **`cache/`**: Local storage for numpy embedding files.
  - **`output/`**: Generated reports and visualizations.
- **`MASTER_TECHNICAL_CHARTER.md`**: The architectural bible of the project.

## 📝 Configuration

Configuration is managed via `settings.yaml` (ensure this file exists and is properly configured with your target entities and probe parameters).

## 🛡️ License
