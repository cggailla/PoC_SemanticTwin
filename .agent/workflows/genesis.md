---
description: Create a new module analysis 
---

# WORKFLOW: GENESIS (New Module Creation)

**Description:** End-to-end process to design, architect, and implement a new analytical module within the Semantic Twin Engine.
**Trigger:** `/genesis`

---

## STEP 1: SCIENTIFIC FEASIBILITY STUDY
**Persona:** Lead Data Scientist & Researcher
**Goal:** Define the mathematical and logical approach to solve the user's request without writing code.

**Instructions:**
1.  **Analyze the Request:** Understand the business question (e.g., "Compare X vs Y", "Detect Z").
2.  **Methodology Selection:**
    * Review available LLM capabilities (Vector Embeddings, Log-probabilities, Completion generation, etc.).
    * *Constraint:* Do NOT rely on pre-conceived notions. Select the most scientifically accurate method for this specific problem.
    * Define the algorithm (e.g., "Cosine similarity between centroid A and B", "Entropy calculation of token distribution").
3.  **Data Requirements:** List the necessary inputs (anchors, prompts, configuration parameters).
4.  **Output:** Present a concise "Scientific Design Proposal" to the user.

> **STOP:** Wait for user validation of the methodology before proceeding.

---

## STEP 2: ARCHITECTURAL DESIGN
**Persona:** Senior Systems Architect
**Goal:** Define the data contracts and integration strategy.

**References:**
* Read Architecture context: @MASTER_TECHNICAL_CHARTER.md
* Read Interface contract: @core/interfaces.py

**Instructions:**
1.  **Module Identity:** Choose a clean, snake_case name for the module (e.g., `adversarial_test`).
2.  **Data Schema (Pydantic):**
    * Draft the `InputConfig` model (what goes into `settings.yaml`).
    * Draft the `OutputResult` model (what goes into the final JSON report).
    * *Constraint:* Ensure output types are JSON-serializable.
3.  **File Structure Plan:**
    * Confirm the creation of `modules/[module_name]/`.
    * List the files to be created (`models.py`, `logic.py`, `__init__.py`).

---

## STEP 3: IMPLEMENTATION & SCAFFOLDING
**Persona:** Senior Python Backend Engineer
**Goal:** Write the production-grade code.

**References:**
* Coding Standards: @.agent/rules/python_stack.md (if available) or standard Python 3.10+ best practices.

**Instructions:**
1.  **Create Directory:** Initialize `modules/[module_name]/`.
2.  **Implement Models:** Write `modules/[module_name]/models.py` using `pydantic`.
3.  **Implement Logic:** Write `modules/[module_name]/logic.py`.
    * **Inheritance:** The class MUST inherit from `BaseProbe` (from `core.interfaces`).
    * **Method:** Implement the `.run(context)` method strictly.
    * **Error Handling:** Wrap external API calls in try/catch blocks using custom exceptions.
    * **Typing:** Use strict Type Hints for every argument and return value.
4.  **Expose Module:** Create `modules/[module_name]/__init__.py` to expose the main class.

---

## STEP 4: INTEGRATION & CONFIGURATION
**Persona:** DevOps & Integrator
**Goal:** Wire the new module into the Orchestrator.

**Instructions:**
1.  **Config Registry:** Propose the YAML lines to add to `config/settings.yaml` (with default values).
2.  **Orchestrator Update:** If the Orchestrator requires manual registration (check `core/orchestrator.py`), indicate where to add the import.
3.  **Sanity Check:** Create a minimal test script `tests/test_[module_name].py` to verify the module loads and executes a "dry run" without crashing.

---

## STEP 5: FINAL REVIEW
**Persona:** Quality Assurance
**Goal:** Verify compliance with the Charter.

**Instructions:**
1.  Check: Does the module import anything from `main.py`? (Forbidden).
2.  Check: Are there any hardcoded values inside `logic.py`? (Forbidden).
3.  Check: Does the `run()` method return the Pydantic Output model defined in Step 2?

**Output:** "Module [Name] is ready for deployment. Please update your `settings.yaml`."e config.