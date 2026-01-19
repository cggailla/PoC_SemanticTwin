---
description: Tune a module based on the results we got
---

# WORKFLOW: TUNE (Data Science Iteration)

**Description:** Systematic process to diagnose, adjust, and optimize the parameters of an existing analysis module when results are unexpected or suboptimal.
**Trigger:** `/tune`

---

## STEP 1: ROOT CAUSE DIAGNOSIS
**Persona:** Lead Data Scientist
**Goal:** Identify WHY the current results are unsatisfactory without changing any code yet.

**References:**
* Read the target Module Logic: @modules/ (target `logic.py`)
* Read the latest Output: (Ask user to paste or point to the JSON result)

**Instructions:**
1.  **Analyze the Symptoms:**
    * Is it a Magnitude issue? (e.g., Score is always 0.99 or 0.00).
    * Is it a Semantic issue? (e.g., "Apple" the fruit is confused with "Apple" the brand).
    * Is it a Stability issue? (e.g., Run 1 != Run 2).
2.  **Audit the Components:**
    * Check the Prompt Engineering (ambiguity, context window).
    * Check the Anchors (are they orthogonal?).
    * Check the Distance Metric (Cosine vs Euclidean vs Jaccard).

> **STOP:** Propose a specific diagnosis to the user (e.g., "The anchors are too synonymous, causing vector collapse"). Wait for validation.

---

## STEP 2: HYPOTHESIS GENERATION (A/B DESIGN)
**Persona:** Experimental Researcher
**Goal:** Define variables to change for the experiment.

**Instructions:**
1.  **Select Variables:** Choose MAX 2 variables to tune (e.g., "Change temperature 0 -> 0.2" AND "Refine system prompt").
2.  **Define Success Criteria:** What does "Better" look like? (e.g., "Drift score should decrease by at least 0.1 for the Legacy test case").

---

## STEP 3: SANDBOX EXPERIMENTATION
**Persona:** Python Scripting Engineer
**Goal:** Run a side-by-side test without breaking the production codebase.

**References:**
* Coding Standards: @.agent/rules/python_stack.md

**Instructions:**
1.  **Create Experiment Script:**
    * Create a temporary file: `experiments/tune_[module_name]_[timestamp].py`.
    * Import the existing `logic.py` class BUT override the method/parameters to test.
    * *Alternative:* Copy the critical function to the experiment script to modify it locally.
2.  **Run Comparison:**
    * Execute the script to print: `[BASELINE RESULT]` vs `[EXPERIMENTAL RESULT]`.
3.  **Review:** Does the change meet the success criteria defined in Step 2?

> **STOP:** Present the side-by-side results to the user. Ask: "Should we apply this fix to Production?"

---

## STEP 4: PRODUCTION PATCHING
**Persona:** Senior Backend Engineer
**Goal:** Apply the validated parameters to the main codebase cleanly.

**Instructions:**
1.  **Modify Source:** Update `modules/[module_name]/logic.py` with the new values/logic.
2.  **Clean Up:**
    * Update docstrings to reflect the change (e.g., "Updated anchors on [Date] to fix bias").
    * Delete or archive the script in `experiments/`.
3.  **Regression Check:** Run the standard test `tests/test_[module_name].py` to ensure no syntax errors were introduced.

**Output:** "Module [Name] successfully tuned. New logic applied to production."