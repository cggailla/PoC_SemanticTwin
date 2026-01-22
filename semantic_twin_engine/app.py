"""
Flask backend API for Semantic Twin Engine.

This module provides REST endpoints for creating and managing semantic twin audits.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import (
    AnchorConfig,
    EntityConfig,
    OpenAIConfig,
    OutputConfig,
    ProbeConfig,
    Settings,
)
from core.job_manager import JobManager, JobStatus
from core.orchestrator import Orchestrator, ProbeRegistry

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize components
job_manager = JobManager(output_base_dir=Path("./output/jobs"))
probe_registry = ProbeRegistry()


def register_probes():
    """Register all available probes."""
    from modules.cluster_validator import ClusterValidator
    from modules.logit_probe import LogitProbe
    from modules.vector_probe import VectorProbe

    probe_registry.register("vector_probe", VectorProbe)
    probe_registry.register("logit_probe", LogitProbe)
    probe_registry.register("cluster_validator", ClusterValidator)

    logger.info("Probes registered: vector_probe, logit_probe, cluster_validator")


def create_settings_from_request(data: Dict[str, Any]) -> Settings:
    """Create Settings object from API request data.

    Args:
        data: Request JSON data

    Returns:
        Settings object

    Raises:
        ValueError: If required fields are missing
    """
    # Accept multiple possible field names from frontend/backends
    entity_name = (data.get("entity_name") or data.get("company_name") or "").strip()
    if not entity_name:
        raise ValueError("entity_name is required")

    # Legacy keywords may be sent as 'legacy_keywords' or 'legacy_identity'
    legacy_keywords = data.get("legacy_keywords", data.get("legacy_identity", []))
    if isinstance(legacy_keywords, str):
        # allow newline- or comma-separated strings
        if "\n" in legacy_keywords:
            legacy_keywords = [
                kw.strip() for kw in legacy_keywords.split("\n") if kw.strip()
            ]
        else:
            legacy_keywords = [
                kw.strip() for kw in legacy_keywords.split(",") if kw.strip()
            ]

    # Strategy keywords may be sent as 'strategy_keywords' or 'strategic_target'
    strategy_keywords = data.get("strategy_keywords", data.get("strategic_target", []))
    if isinstance(strategy_keywords, str):
        if "\n" in strategy_keywords:
            strategy_keywords = [
                kw.strip() for kw in strategy_keywords.split("\n") if kw.strip()
            ]
        else:
            strategy_keywords = [
                kw.strip() for kw in strategy_keywords.split(",") if kw.strip()
            ]

    if not legacy_keywords or not strategy_keywords:
        raise ValueError("Both legacy_keywords and strategy_keywords are required")

    # Create entity config
    entity_config = EntityConfig(
        name=entity_name,
        legacy_identity=legacy_keywords,
        strategic_target=strategy_keywords,
    )

    # Create anchor config (optional)
    anchor_config = AnchorConfig(
        entities=data.get("anchor_entities", []),
        concepts=data.get("anchor_concepts", []),
    )

    # Create probe configs
    probe_configs = {
        "vector_probe": ProbeConfig(
            enabled=data.get("enable_vector_probe", True),
            params=data.get(
                "vector_probe_params",
                {
                    "embedding_dimensions": 1536,
                    "top_k_anchors": 5,
                    "dimensions": {
                        "default": {
                            "contextual_prompt": f"The {{{entity_name}}} is characterized by",
                            "anchor_a": legacy_keywords,
                            "anchor_b": strategy_keywords,
                        }
                    },
                },
            ),
        ),
        "cluster_validator": ProbeConfig(
            enabled=data.get("enable_cluster_validator", True),
            params=data.get(
                "cluster_validator_params",
                {
                    "thresholds": {
                        "z_score_threshold": 2.0,
                        "min_density": 0.60,
                        "min_separation": 0.25,
                        "max_outliers": 3,
                    }
                },
            ),
        ),
        "logit_probe": ProbeConfig(enabled=False),
    }

    # Create OpenAI config
    openai_config = OpenAIConfig(
        model=data.get("model", "gpt-4o"),
        embedding_model=data.get("embedding_model", "text-embedding-3-small"),
        temperature=0.0,  # Always deterministic
        seed=42,
        max_retries=3,
    )

    # Create output config
    output_config = OutputConfig(
        directory=Path("./output"),
        format="json",
        include_metadata=True,
    )

    # Create and return Settings
    settings = Settings(
        entity=entity_config,
        anchors=anchor_config,
        probes=probe_configs,
        openai=openai_config,
        output=output_config,
    )

    return settings


def execute_audit(job_id: str, settings: Settings, output_dir: Path):
    """Execute audit in background.

    Args:
        job_id: Job identifier
        settings: Settings configuration
        output_dir: Output directory for results
    """
    try:
        logger.info(f"Executing audit for job {job_id}: {settings.entity.name}")

        # Create orchestrator
        orchestrator = Orchestrator(settings=settings, registry=probe_registry)

        # Run audit
        audit_report = orchestrator.run()

        # Save report
        report_path = (
            output_dir / f"audit_{settings.entity.name.lower().replace(' ', '_')}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Finalize report
        audit_report.finalize()

        # Save as JSON
        report_dict = audit_report.to_dict(include_metadata=True)
        # resolve to absolute paths to avoid later cwd-dependent mismatches
        report_path = report_path.resolve()
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Try to generate HTML visual report if visualizer is available
        try:
            from modules.reporting.visualizer import AuditReportVisualizer

            visualizer = AuditReportVisualizer()
            # The visualizer expects the vector_probe data and dimensions config
            vector_data = (
                report_dict.get("probes", {}).get("vector_probe", {}).get("data", {})
            )
            dimensions_config = (
                settings.probes.get("vector_probe", {}).params.get("dimensions", {})
                if settings.probes.get("vector_probe")
                else {}
            )

            report_html_path = (
                output_dir
                / f"report_{settings.entity.name.lower().replace(' ', '_')}.html"
            )
            # ensure the visualizer writes to resolved absolute path
            report_html_path = report_html_path.resolve()
            visualizer.generate_report(
                vector_data, dimensions_config, settings.entity.name, report_html_path
            )
            job_manager.set_job_metadata(
                job_id, {"visual_report": str(report_html_path)}
            )
        except Exception:
            logger.exception("Failed to generate visual report")

        job_manager.set_job_output(job_id, str(report_path))
        job_manager.set_job_metadata(
            job_id,
            {
                "entity_name": settings.entity.name,
                "probes_succeeded": audit_report.metadata.get("probes_succeeded", 0),
            },
        )

        logger.info(f"Audit completed successfully for job {job_id}")

    except Exception:
        logger.exception(f"Audit failed for job {job_id}")
        raise


# ============================================================================
# API Routes
# ============================================================================


@app.route("/", methods=["GET"])
def index():
    """Serve landing page."""
    repo_root = Path(__file__).resolve().parent.parent
    index_path = repo_root / "index.html"
    if not index_path.exists():
        return jsonify(
            {
                "error": "index.html not found",
                "expected_path": str(index_path),
            }
        ), 404
    return send_file(str(index_path), mimetype="text/html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "Semantic Twin Engine",
        }
    ), 200


@app.route("/api/audit", methods=["POST"])
def create_audit():
    """Create and start a new audit job.

    Request JSON:
        - entity_name: str (required)
        - legacy_keywords: list[str] or str (comma-separated)
        - strategy_keywords: list[str] or str (comma-separated)
        - anchor_entities: list[str] (optional)
        - anchor_concepts: list[str] (optional)
        - model: str (optional, default: "gpt-4o")
        - embedding_model: str (optional, default: "text-embedding-3-small")
        - enable_vector_probe: bool (optional, default: True)
        - enable_cluster_validator: bool (optional, default: True)

    Returns:
        JSON with job_id and status
    """
    try:
        # Diagnostic logging: headers, content-type, raw body, and form data
        try:
            headers = {k: v for k, v in request.headers.items()}
        except Exception:
            headers = str(request.headers)

        logger.info("/api/audit called: content_type=%s", request.content_type)
        logger.debug("/api/audit request headers: %s", headers)

        # Read raw body early (safe to call multiple times in Flask)
        raw_body = request.get_data(as_text=True)
        logger.debug("/api/audit raw body (truncated 2000 chars): %s", raw_body[:2000])
        logger.debug("/api/audit form data: %s", request.form.to_dict())

        # Try JSON body first (silent to avoid exceptions)
        data = request.get_json(silent=True)
        logger.debug("/api/audit parsed JSON (silent): %s", data)

        # If no JSON, accept form-encoded bodies
        if data is None:
            if request.form:
                data = request.form.to_dict()
                logger.debug("/api/audit using form data: %s", data)
            else:
                # Try to parse raw body as JSON as a last resort
                try:
                    data = json.loads(raw_body) if raw_body else None
                    logger.debug("/api/audit parsed JSON (raw): %s", data)
                except Exception as e:
                    logger.debug("/api/audit failed to parse raw body as JSON: %s", e)
                    data = None

        if not data:
            logger.warning("Empty or invalid request body received for /api/audit")
            return (
                jsonify({"error": "Empty request body or invalid JSON/form data"}),
                400,
            )

        # Validate and create settings
        logger.info(
            "/api/audit creating settings from request data for entity: %s",
            data.get("entity_name"),
        )
        settings = create_settings_from_request(data)

        # Create job
        job_id = job_manager.create_job()
        output_dir = job_manager.get_job_output_dir(job_id)

        # Execute audit in background
        job_manager.execute_job(job_id, execute_audit, job_id, settings, output_dir)

        return jsonify(
            {
                "job_id": job_id,
                "status": JobStatus.PENDING.value,
                "entity_name": settings.entity.name,
            }
        ), 202

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Error creating audit")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get job status and metadata.

    Returns:
        JSON with job status and metadata
    """
    try:
        job_result = job_manager.get_job_status(job_id)

        if not job_result:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(job_result.to_dict()), 200

    except Exception:
        logger.exception("Error getting job status")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/results/<job_id>", methods=["GET"])
def get_job_results(job_id: str):
    """Get job results JSON.

    Returns:
        JSON report from the audit
    """
    try:
        job_result = job_manager.get_job_status(job_id)

        if not job_result:
            return jsonify({"error": "Job not found"}), 404

        if job_result.status != JobStatus.COMPLETED:
            return jsonify(
                {"error": f"Job not completed (status: {job_result.status.value})"}
            ), 400

        output_path = job_result.output_path
        if not output_path or not Path(output_path).exists():
            return jsonify({"error": "Results not found"}), 404

        with open(output_path, "r") as f:
            results = json.load(f)

        return jsonify(results), 200

    except Exception:
        logger.exception("Error getting job results")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/report/<job_id>", methods=["GET"])
def get_job_report(job_id: str):
    """Get job HTML report (if available).

    Returns:
        HTML report file
    """
    try:
        job_result = job_manager.get_job_status(job_id)

        if not job_result:
            return jsonify({"error": "Job not found"}), 404

        if job_result.status != JobStatus.COMPLETED:
            return jsonify(
                {"error": f"Job not completed (status: {job_result.status.value})"}
            ), 400

        # 1) Check for explicit visual_report path in job metadata
        visual_path = None
        try:
            visual_path = job_result.metadata.get("visual_report")
        except Exception:
            visual_path = None

        repo_root = Path(__file__).resolve().parent.parent

        # Helper: try a path and return first HTML file found inside or the file itself
        def find_html_in_path(p: Path):
            try:
                if p.is_file() and p.exists():
                    return p
                if p.is_dir():
                    files = list(p.glob("*.html"))
                    if files:
                        return files[0]
            except Exception:
                return None
            return None

        # If metadata contains a path, check it directly
        if visual_path:
            vp = Path(visual_path)
            found = find_html_in_path(vp)
            if found:
                return send_file(str(found), mimetype="text/html")

        # Candidate locations to search (in order)
        candidates = []

        # Job manager output dir (resolved)
        try:
            jm_dir = job_manager.get_job_output_dir(job_id)
            candidates.append(jm_dir)
        except Exception:
            pass

        # Repo-root output/jobs/<job_id>
        candidates.append(repo_root / "output" / "jobs" / job_id)

        # semantic_twin_engine/output/jobs/<job_id> (legacy)
        candidates.append(
            repo_root / "semantic_twin_engine" / "output" / "jobs" / job_id
        )

        # Global visuals folder
        candidates.append(repo_root / "semantic_twin_engine" / "output" / "visuals")
        candidates.append(repo_root / "output" / "visuals")

        # Search candidates
        for cand in candidates:
            found = find_html_in_path(cand)
            if found:
                return send_file(str(found), mimetype="text/html")

        # As a last resort, search workspace for html files containing job id
        try:
            for p in repo_root.rglob(f"*{job_id}*.html"):
                return send_file(str(p), mimetype="text/html")
        except Exception:
            pass

        return jsonify({"error": "HTML report not found"}), 404

    except Exception:
        logger.exception("Error getting job report")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all jobs.

    Returns:
        JSON list of all jobs with their status
    """
    try:
        jobs = []
        for job_id, job_result in job_manager._jobs.items():
            jobs.append(job_result.to_dict())

        return jsonify({"jobs": jobs, "total": len(jobs)}), 200

    except Exception:
        logger.exception("Error listing jobs")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/generate_report/<job_id>", methods=["POST"])
def generate_report_for_job(job_id: str):
    """Generate visual HTML report for an existing job's JSON output.

    This is useful to create the visual dashboard after the audit finished
    if it wasn't generated automatically.
    """
    try:
        # Locate JSON report in job output dir
        job_output_dir = job_manager.get_job_output_dir(job_id)
        json_files = list(job_output_dir.glob("audit_*.json"))
        if not json_files:
            return jsonify({"error": "Audit JSON not found for job"}), 404

        report_json_path = json_files[0]
        with open(report_json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Prepare visualizer inputs
        try:
            from modules.reporting.visualizer import AuditReportVisualizer
        except Exception:
            logger.exception("Visualizer import failed")
            return jsonify({"error": "Visualizer not available"}), 500

        visualizer = AuditReportVisualizer()
        vector_data = (
            report_data.get("probes", {}).get("vector_probe", {}).get("data", {})
        )
        dimensions_config = (
            request.json.get("dimensions_config")
            if request.json and "dimensions_config" in request.json
            else {}
        )

        out_path = job_output_dir / f"report_{report_data.get('entity', 'audit')}.html"
        visualizer.generate_report(
            vector_data, dimensions_config, report_data.get("entity", ""), out_path
        )
        job_manager.set_job_metadata(job_id, {"visual_report": str(out_path)})

        return jsonify({"report_path": str(out_path)}), 200

    except Exception:
        logger.exception("Error generating report for job %s", job_id)
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Error Handlers
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Application Initialization
# ============================================================================


def create_app():
    """Create and configure Flask application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Register probes
    register_probes()

    logger.info("Flask application initialized")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
