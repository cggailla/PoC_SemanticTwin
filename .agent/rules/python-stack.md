---
trigger: always_on
---

# PYTHON CODING STANDARDS & STACK

**Context:** This project is a Scientific Audit Engine. Precision, reproducibility, and type safety are paramount.

## 1. APPROVED TECH STACK
* **Language:** Python 3.10+
* **Data Validation:** `pydantic` V2 (Strict Mode preferred).
* **AI Integration:** `openai` (Latest SDK).
* **Math/Vectors:** `numpy` (for vector ops), `scikit-learn` (for PCA, cosine similarity).
* **Resilience:** `tenacity` (for API retries and backoff).
* **Configuration:** `python-dotenv` & `pyyaml`.
* **Caching:** `numpy` .npz files (for embedding persistence).
* **Visualization:** `plotly` (for interactive charts).
* **Testing:** `pytest`.

## 2. STRICT GUIDELINES

### A. Data Structures & Typing
1.  **No Raw Dicts:** NEVER pass `dict` or `json` objects between functions. Always define and pass a **Pydantic Model**.
    * *Bad:* `def process(data: dict):`
    * *Good:* `def process(data: AuditInput):`
2.  **Strict Type Hints:** Every function signature must be fully typed. Use `typing.List`, `typing.Optional`, `typing.Dict`.
3.  **Immutable Defaults:** Never use mutable default arguments (e.g., `list=[]`). Use `None` instead.

### B. Scientific Rigor (Math & Vectors)
1.  **Numpy First:** Do not use Python lists for vector math. Convert embeddings to `np.array` immediately upon reception.
2.  **Precision:** Round floats to 4 decimal places ONLY in the final Output layer (JSON). Keep full precision during internal calculations.

### C. Operational Resilience
1.  **API Safety:** Wrap ALL external API calls (OpenAI) with the `@retry` decorator from `tenacity`.
    * *Config:* `stop=stop_after_attempt(3)`, `wait=wait_exponential(multiplier=1, min=4, max=10)`.
2.  **Path Handling:** Use `pathlib.Path` exclusively. Do not use `os.path.join` or string concatenation for paths.

### D. Embedding Cache Strategy
1.  **Use EmbeddingStore:** Always use `core.embedding_store.EmbeddingStore` for embedding operations.
2.  **Cache Location:** All embeddings are cached in `./cache/` as NumPy .npz files.
3.  **Hash Keys:** Use SHA256(model + text) for cache keys to ensure collision-free storage.
4.  **Auto-Save:** The orchestrator auto-saves cache after each run. For standalone scripts, call `store.save()` explicitly.

### E. Code Style & Docs
1.  **Docstrings:** Google Style is MANDATORY for all classes and public methods.
    * Must include: `Args:`, `Returns:`, and `Raises:`.
2.  **Imports:** Group imports: Standard Lib -> Third Party -> Local Application. Use absolute imports for local modules (e.g., `from core.interfaces import BaseProbe`).

## 3. ERROR HANDLING STRATEGY
* **Custom Exceptions:** Only raise exceptions defined in `core/exceptions.py`.
* **No Silent Failures:** Never use bare `try: ... except: pass`. Log the error and re-raise or handle explicitly.