"""
Audit Report Visualizer for the Semantic Twin Engine.

Generates a unified HTML report with:
1. Bar chart summary of drift scores by dimension
2. Tabbed semantic axis plots for each dimension
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import yaml

from modules.base_probe import BaseProbe

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "legacy": "#8B4513",
    "legacy_light": "rgba(139, 69, 19, 0.5)",
    "strategy": "#2E8B57",
    "strategy_light": "rgba(46, 139, 87, 0.5)",
    "entity": "#1E90FF",
    "neutral": "#666666",
    "axis": "#333333",
    "positive": "#2E8B57",
    "negative": "#8B4513",
}


class AuditReportVisualizer(BaseProbe):
    """Generates visual reports from audit results.
    
    Creates a unified HTML dashboard with:
    - Summary bar chart of drift scores
    - Individual semantic axis plots for each dimension
    """
    
    @property
    def name(self) -> str:
        return "report_visualizer"
    
    def run(self, context):
        """Not used - this class generates reports from completed audits."""
        pass
    
    def generate_report(
        self,
        audit_data: dict[str, Any],
        dimensions_config: dict[str, Any],
        entity_name: str,
        output_path: Path,
    ) -> Path:
        """Generate complete visual report from audit data.
        
        Args:
            audit_data: The vector_probe results from audit JSON.
            dimensions_config: Dimension configs from settings.yaml.
            entity_name: Name of the entity.
            output_path: Where to save the HTML report.
        
        Returns:
            Path to the generated HTML file.
        """
        logger.info("Generating visual report for %s", entity_name)
        
        # Extract dimension results
        dimensions = audit_data.get("dimensions", {})
        
        # Generate bar chart
        bar_chart = self._create_bar_chart(dimensions, entity_name)
        
        # Generate axis plots for each dimension
        axis_plots = []
        for dim_name, dim_result in dimensions.items():
            if dim_name in dimensions_config:
                config = dimensions_config[dim_name]
                axis_plot, anchors_html = self._create_axis_plot(
                    dimension_name=dim_name,
                    anchor_a=config["anchor_a"],
                    anchor_b=config["anchor_b"],
                    contextual_prompt=config["contextual_prompt"],
                    entity_name=entity_name,
                    drift_score=dim_result["drift_score"],
                )
                axis_plots.append((dim_name, (axis_plot, anchors_html)))
        
        # Combine into HTML with tabs
        html_content = self._build_html_report(
            entity_name=entity_name,
            bar_chart=bar_chart,
            axis_plots=axis_plots,
            audit_data=audit_data,
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info("Report saved to %s", output_path)
        return output_path
    
    def _create_bar_chart(
        self,
        dimensions: dict[str, Any],
        entity_name: str,
    ) -> str:
        """Create horizontal bar chart of drift scores.
        
        Args:
            dimensions: Dictionary of dimension results.
            entity_name: Name of the entity.
        
        Returns:
            Plotly figure as HTML div.
        """
        # Sort by drift score
        sorted_dims = sorted(
            dimensions.items(),
            key=lambda x: x[1]["drift_score"],
            reverse=True
        )
        
        names = [d[0].replace("_", " ").title() for d in sorted_dims]
        scores = [d[1]["drift_score"] for d in sorted_dims]
        colors = [COLORS["positive"] if s >= 0 else COLORS["negative"] for s in scores]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=names,
            x=scores,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:+.2f}" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Drift Score: %{x:+.4f}<extra></extra>",
        ))
        
        # Add zero line
        fig.add_vline(x=0, line_width=2, line_color="black")
        
        fig.update_layout(
            title=dict(
                text=f"<b>Semantic Positioning: {entity_name}</b>",
                x=0.5,
                font=dict(size=18),
            ),
            xaxis=dict(
                title="Drift Score",
                range=[-1.2, 1.2],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["LEGACY<br>(-1)", "-0.5", "NEUTRAL<br>(0)", "+0.5", "STRATEGY<br>(+1)"],
                gridcolor="#E0E0E0",
            ),
            yaxis=dict(title=""),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            height=max(300, len(dimensions) * 60 + 100),
            margin=dict(l=150, r=80, t=80, b=60),
        )
        
        return fig.to_html(include_plotlyjs=False, full_html=False, div_id="bar_chart")
    
    def _create_axis_plot(
        self,
        dimension_name: str,
        anchor_a: list[str],
        anchor_b: list[str],
        contextual_prompt: str,
        entity_name: str,
        drift_score: float,
    ) -> str:
        """Create 1D axis plot for a dimension.
        
        Args:
            dimension_name: Name of the dimension.
            anchor_a: Legacy anchor terms.
            anchor_b: Strategy anchor terms.
            contextual_prompt: Prompt template.
            entity_name: Entity name.
            drift_score: Pre-calculated drift score.
        
        Returns:
            Plotly figure as HTML div.
        """
        # Get embeddings (uses cache)
        embeddings_a = np.array(self.get_embeddings(anchor_a))
        embeddings_b = np.array(self.get_embeddings(anchor_b))
        
        prompt = contextual_prompt.format(entity=entity_name)
        entity_embedding = np.array(self.get_embedding(prompt))
        
        # Compute centroids
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # PCA to 1D
        all_embeddings = np.vstack([
            embeddings_a, embeddings_b,
            [centroid_a], [centroid_b], [entity_embedding]
        ])
        
        pca = PCA(n_components=1)
        embeddings_1d = pca.fit_transform(all_embeddings).flatten()
        
        n_a, n_b = len(anchor_a), len(anchor_b)
        points_a = embeddings_1d[:n_a]
        points_b = embeddings_1d[n_a:n_a + n_b]
        centroid_a_1d = embeddings_1d[n_a + n_b]
        centroid_b_1d = embeddings_1d[n_a + n_b + 1]
        entity_1d = embeddings_1d[n_a + n_b + 2]
        
        if centroid_a_1d > centroid_b_1d:
            points_a = -points_a
            points_b = -points_b
            centroid_a_1d = -centroid_a_1d
            centroid_b_1d = -centroid_b_1d
            entity_1d = -entity_1d
            
        # Normalize: Legacy=-1, Strategy=+1
        midpoint = (centroid_a_1d + centroid_b_1d) / 2
        half_range = abs(centroid_b_1d - centroid_a_1d) / 2
        
        if half_range > 0:
            norm_points_a = (points_a - midpoint) / half_range
            norm_points_b = (points_b - midpoint) / half_range
            norm_entity = (entity_1d - midpoint) / half_range
        else:
            norm_points_a = np.zeros_like(points_a)
            norm_points_b = np.zeros_like(points_b)
            norm_entity = 0
        
        # Create figure
        fig = go.Figure()
        
        # Axis line
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[0, 0],
            mode="lines",
            line=dict(color=COLORS["axis"], width=3),
            showlegend=False, hoverinfo="skip",
        ))
        
        # Legacy points
        jitter_a = np.random.uniform(-0.12, -0.04, n_a)
        fig.add_trace(go.Scatter(
            x=norm_points_a, y=jitter_a,
            mode="markers",
            marker=dict(size=7, color=COLORS["legacy"], opacity=0.6),
            name="Legacy",
            text=anchor_a,
            hovertemplate="<b>%{text}</b><extra>Legacy</extra>",
        ))
        
        # Strategy points
        jitter_b = np.random.uniform(0.04, 0.12, n_b)
        fig.add_trace(go.Scatter(
            x=norm_points_b, y=jitter_b,
            mode="markers",
            marker=dict(size=7, color=COLORS["strategy"], opacity=0.6),
            name="Strategy",
            text=anchor_b,
            hovertemplate="<b>%{text}</b><extra>Strategy</extra>",
        ))
        
        # Centroids
        fig.add_trace(go.Scatter(
            x=[-1], y=[0],
            mode="markers",
            marker=dict(size=20, color=COLORS["legacy"], symbol="diamond", line=dict(color="white", width=2)),
            name="Legacy Centroid",
            hovertemplate="<b>Legacy Centroid</b><extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[1], y=[0],
            mode="markers",
            marker=dict(size=20, color=COLORS["strategy"], symbol="diamond", line=dict(color="white", width=2)),
            name="Strategy Centroid",
            hovertemplate="<b>Strategy Centroid</b><extra></extra>",
        ))
        
        # Entity
        entity_color = COLORS["positive"] if drift_score >= 0 else COLORS["negative"]
        fig.add_trace(go.Scatter(
            x=[norm_entity], y=[0],
            mode="markers+text",
            marker=dict(size=25, color=COLORS["entity"], symbol="star", line=dict(color="white", width=2)),
            text=[entity_name],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["entity"]),
            name=entity_name,
            hovertemplate=f"<b>{entity_name}</b><br>Score: {drift_score:+.4f}<extra></extra>",
        ))
        
        dim_display = dimension_name.replace("_", " ").title()
        variance = pca.explained_variance_ratio_[0] * 100
        
        fig.update_layout(
            title=dict(
                text=f"<b>{dim_display}</b> <span style='font-size:12px'>({variance:.0f}% var.)</span>",
                x=0.5,
                font=dict(size=14),
            ),
            xaxis=dict(
                range=[-1.5, 1.5],
                tickvals=[-1, 0, 1],
                ticktext=["LEGACY", "0", "STRATEGY"],
                tickfont=dict(size=10),
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=[-0.25, 0.25],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            showlegend=False,
            height=200,
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        div_id = f"axis_{dimension_name}"
        plot_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id)
        
        # Generate anchor lists HTML
        def format_anchors(anchors):
            return "<ul style='list-style-type: none; padding: 0; column-count: 2; column-gap: 20px;'>" + \
                   "".join(f"<li style='padding: 2px 0; font-size: 12px; color: #444;'>• {term}</li>" for term in anchors) + \
                   "</ul>"

        anchors_html = {
            "legacy": format_anchors(anchor_a),
            "strategy": format_anchors(anchor_b)
        }
        
        return plot_html, anchors_html
    
    def _build_html_report(
        self,
        entity_name: str,
        bar_chart: str,
        axis_plots: list[tuple[str, str]],
        audit_data: dict[str, Any],
    ) -> str:
        """Build complete HTML report with tabs.
        
        Args:
            entity_name: Entity name.
            bar_chart: Bar chart HTML.
            axis_plots: List of (name, html) tuples.
            audit_data: Raw audit data.
        
        Returns:
            Complete HTML string.
        """
        # Build tab buttons
        tab_buttons = []
        tab_buttons.append('<button class="tab-btn active" onclick="showTab(\'summary\')">Summary</button>')
        for dim_name, _ in axis_plots:
            display_name = dim_name.replace("_", " ").title()
            tab_buttons.append(f'<button class="tab-btn" onclick="showTab(\'{dim_name}\')">{display_name}</button>')
        
        # Build tab contents
        tab_contents = []
        tab_contents.append(f'<div id="tab-summary" class="tab-content active">{bar_chart}</div>')
        for dim_name, (plot_html, anchors_html) in axis_plots:
            tab_contents.append(f'''
            <div id="tab-{dim_name}" class="tab-content">
                {plot_html}
                <div style="display: flex; gap: 40px; margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 8px;">
                    <div style="flex: 1;">
                        <h3 style="color: #8B4513; border-bottom: 2px solid #8B4513; padding-bottom: 5px; margin-bottom: 10px; font-size: 14px;">Legacy Anchors</h3>
                        {anchors_html['legacy']}
                    </div>
                    <div style="flex: 1;">
                        <h3 style="color: #2E8B57; border-bottom: 2px solid #2E8B57; padding-bottom: 5px; margin-bottom: 10px; font-size: 14px;">Strategy Anchors</h3>
                        {anchors_html['strategy']}
                    </div>
                </div>
            </div>
            ''')
        
        # Summary stats
        avg_score = audit_data.get("summary", "").split(":")[-1].strip() if "summary" in audit_data else "N/A"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Semantic Twin Report: {entity_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tab-btn {{ padding: 10px 20px; border: none; background: #e0e0e0; border-radius: 6px; cursor: pointer; font-size: 13px; transition: all 0.2s; }}
        .tab-btn:hover {{ background: #d0d0d0; }}
        .tab-btn.active {{ background: #1E90FF; color: white; }}
        .tab-content {{ display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tab-content.active {{ display: block; }}
        .legend {{ display: flex; gap: 30px; justify-content: center; margin-top: 15px; font-size: 13px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Semantic Twin Report</h1>
        <div class="subtitle">{entity_name}</div>
    </div>
    
    <div class="container">
        <div class="tabs">
            {''.join(tab_buttons)}
        </div>
        
        {''.join(tab_contents)}
        
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#8B4513"></div> Legacy</div>
            <div class="legend-item"><div class="legend-dot" style="background:#2E8B57"></div> Strategy</div>
            <div class="legend-item"><div class="legend-dot" style="background:#1E90FF"></div> Entity Position</div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');
            // Trigger Plotly resize
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>"""
        
        return html
