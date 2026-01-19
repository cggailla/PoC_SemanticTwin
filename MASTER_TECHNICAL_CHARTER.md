# MASTER TECHNICAL CHARTER: Semantic Twin Engine

**Project:** Semantic Twin Engine (STE)
**Type:** SaaS Backend / Audit Engine
**Version:** 1.0
**Target Audience:** AI Developer / Senior Software Engineer

---

## 1. Project Context & Philosophy

The **Semantic Twin Engine** is a deterministic system designed to probe, map, and quantify the latent representation of corporate entities within Large Language Models (LLMs).

* **Core Philosophy:** We treat LLMs as **probabilistic databases**, not chatbots. We probe weights, vectors, and logits.
* **Goal:** Produce an auditable, scientific report (JSON) measuring the distance between a company's "Legacy Identity" and its "Strategic Target".
* **State:** This is a Production-Grade PoC. It must be architected as a scalable backend micro-service, not a data science script.

---

## 2. Architecture: The "Orchestrator & Plugin" Pattern

The system is strictly divided into **Core** (Stability) and **Modules** (Business Logic).

### 2.1 Directory Structure

The codebase **must** strictly follow this hierarchy:

```text
/semantic_twin_engine
│
├── /core                  # THE INFRASTRUCTURE (Immutable logic)
│   ├── orchestrator.py    # Main pipeline runner (Loads config -> Runs Probes -> Saves)
│   ├── interfaces.py      # Abstract Base Classes (The Contract)
│   ├── config_loader.py   # Pydantic models for configuration
│   └── data_manager.py    # Handles I/O and result aggregation
│
├── /modules               # THE LOGIC (Pluggable Probes)
│   ├── base_probe.py      # Parent class for all probes
│   │
│   ├── /vector_probe      # Module A: Embeddings & Geometry
│   │   └── logic.py
│   │
│   └── /logit_probe       # Module B: Logprobs & Subconscious
│       └── logic.py
│
├── /config                # CONFIGURATION
│   └── settings.yaml      # User inputs (Entity name, Anchors)
│
└── main.py                # Entry Point

```

### 2.2 The "Contract" (Interface Rule)

Every analytical module MUST inherit from `BaseProbe` and implement the `.run(context)` method. The Orchestrator does not know the internal logic of a probe; it only calls `.run()`.

---

## 3. Coding Standards (The "Do's")

### 3.1 Syntax & Typing

* **Language:** Python 3.10+
* **Type Hinting:** **MANDATORY**. Every function signature must have type hints.
* *Bad:* `def get_vector(text):`
* *Good:* `def get_vector(text: str) -> np.ndarray:`


* **Docstrings:** All classes and public methods must have Google-style docstrings explaining Args, Returns, and Raises.

### 3.2 Data Validation

* Use **Pydantic** for all data structures (Configs, Inputs, Outputs).
* Do not pass raw Dictionaries around. Use structured Objects.

### 3.3 Error Handling

* **No Silent Failures:** Never use bare `try/except: pass`.
* **Graceful Degradation:** If one module fails (e.g., API timeout), catch the custom exception, log the error in the report, and allow the Orchestrator to proceed to the next module.

---

## 4. Anti-Patterns (The "Don'ts")

### 4.1 NO Hardcoding

* **Strictly Forbidden:** Writing entity names (e.g., "Philip Morris"), specific anchors, or magic numbers inside Python files.
* **Solution:** All variable data comes from `settings.yaml` or `.env`.

### 4.2 NO "Notebook Style"

* Do not write long procedural scripts.
* Do not use global variables.
* Do not use `print()` for debugging in production code (use a `logger`).

### 4.3 NO Chat Logic in Core

* Do not simulate conversations. We use the API for **Completion** (Logprobs) or **Embeddings**.
* Keep the system deterministic (set `temperature=0` and `seed` where applicable).

---

## 5. Development Workflow for the AI

When asked to implement a feature, follow this sequence:

1. **Check Interfaces:** Does this require modifying the `BaseProbe` contract? (Avoid if possible).
2. **Define Pydantic Model:** Define the Input/Output schema of the new module.
3. **Implement Logic:** Write the code in `/modules/{module_name}/logic.py`.
4. **Register Module:** Add the module key to `settings.yaml` and the Orchestrator registry.
5. **Integration:** Ensure `main.py` runs the new module without errors.

---

## 6. Technology Stack

* **Core:** `openai`, `pydantic`, `pyyaml`, `python-dotenv`
* **Math:** `numpy`, `scikit-learn`
* **Utils:** `tenacity` (for API retries)

---

**END OF CHARTER**