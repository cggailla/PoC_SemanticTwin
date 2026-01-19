---
description: Visualize a json data type as a consulting graph
---

# WORKFLOW: VISUALIZE (Report Generation)

**Description:** Process to transform raw audit data (JSON) into high-value, corporate-grade visual artifacts.
**Trigger:** `/visualize`

---

## STEP 1: VISUAL STRATEGY & DATA STORYTELLING
**Persona:** Information Designer & Strategy Consultant
**Goal:** Determine the most effective way to represent the data to answer the client's business question.

**Instructions:**
1.  **Analyze the Data Source:**
    * Read the structure of the provided JSON output (or the `OutputResult` schema from the module).
    * Identify the variables (Continuous? Categorical? Probabilistic?).
2.  **Select Representation:**
    * Determine the optimal chart type (e.g., Radar for multi-axis comparison, Scatter for semantic distance, Heatmap for probability distribution).
    * *Constraint:* Avoid "flashy" or complex charts. Prioritize clarity, legibility, and scientific accuracy.
3.  **Define the Insight:**
    * What is the specific takeaway? (e.g., "Show the gap between current state and target state").
    * Propose the axis labels and title.

> **STOP:** Briefly propose the visualization strategy to the user (e.g., "I recommend a Radar Chart to show the skew between Legacy and Strategy"). Wait for validation.

---

## STEP 2: DATA PREPARATION ARCHITECTURE
**Persona:** Data Engineer
**Goal:** Structure the raw JSON into a plot-ready format (DataFrame/Array).

**References:**
* Read Data Schemas: @modules/ (look for relevant `models.py`)

**Instructions:**
1.  **Ingestion Logic:**
    * Define a function that accepts the Pydantic `AuditResult` or the raw JSON dictionary.
    * Plan the extraction of nested metrics (do not hardcode keys if a schema is available).
2.  **Normalization:**
    * Decide how to handle scales (e.g., -1 to 1 vs 0 to 100%).
    * Plan for edge cases (missing data, null vectors).

---

## STEP 3: VISUAL IMPLEMENTATION
**Persona:** Frontend/Viz Developer
**Goal:** Code the visualization using a Python plotting library (Plotly Graph Objects recommended for interactivity).

**References:**
* Coding Standards: @.agent/rules/python_stack.md

**Instructions:**
1.  **File Location:** Create or update `modules/reporting/visualizer.py` (or a specific reporting module).
2.  **Styling Guidelines (Corporate/Premium):**
    * **Palette:** Use professional, neutral tones with one accent color for the "Target/Strategy" and a muted color for "Legacy".
    * **Layout:** minimalist background, clear grid lines, strictly legible font sizes.
    * **Interactivity:** Ensure tooltips display the exact values/probabilities.
3.  **Code Structure:**
    * The function MUST return the Figure object or an HTML string.
    * It must NOT show (`.show()`) the plot directly (blocking execution), but save it or return it.

---

## STEP 4: ARTIFACT GENERATION & EXPORT
**Persona:** DevOps
**Goal:** Save the result cleanly for the end-user.

**Instructions:**
1.  **Output Management:**
    * Target directory: `outputs/visuals/`.
    * Naming convention: `[Entity_Name]_[Analysis_Type]_[Timestamp].html`.
2.  **Integration:**
    * Create a simple entry point script (e.g., `generate_report.py`) that loads the latest JSON and runs the visualizer.

---

## STEP 5: UX & INTEGRITY CHECK
**Persona:** UX Reviewer
**Goal:** Ensure the chart is not misleading.

**Instructions:**
1.  **Check Scales:** Are axes truncated to exaggerate differences? (Forbidden).
2.  **Check Context:** Does the chart have a clear Title and Legend?
3.  **Check Accessibility:** Are colors distinguishable for colorblind users (e.g., avoid Red/Green only)?

**Output:** "Visualization generated at `outputs/visuals/...`. It depicts [Insight] using a [Chart Type]."