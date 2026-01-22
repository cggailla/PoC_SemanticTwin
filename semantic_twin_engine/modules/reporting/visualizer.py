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
import plotly.offline
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import yaml

from modules.base_probe import BaseProbe

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "legacy": "#8B4513",       # Saddle brown
    "legacy_light": "rgba(139, 69, 19, 0.3)",
    "strategy": "#2E8B57",     # Sea green
    "strategy_light": "rgba(46, 139, 87, 0.3)",
    "entity": "#1E90FF",       # Dodger blue
    "neutral": "#666666",
    "axis": "#333333",
    "positive": "#2E8B57",
    "negative": "#8B4513",
    "centroid_legacy": "#CD853F",   # Peru (darker legacy)
    "centroid_strategy": "#228B22",  # Forest green (darker strategy)
    "comparator_legacy": "#CD853F",   # Match Legacy Centroid (Peru)
    "comparator_strategic": "#228B22", # Match Strategy Centroid (Forest Green)
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
        comparators: dict[str, list[str]] | None = None,
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
                    comparators=comparators or [],
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
        comparators: dict[str, list[str]],
        drift_score: float,
    ) -> tuple[str, dict]:
        """Create 1D axis plot for a dimension.
        
        Args:
            dimension_name: Name of the dimension.
            anchor_a: Legacy anchor terms.
            anchor_b: Strategy anchor terms.
            contextual_prompt: Prompt template.
            entity_name: Entity name.
            drift_score: Pre-calculated drift score (used for validation).
        
        Returns:
            Tuple of (Plotly figure as HTML div, Dictionary of anchor HTML lists).
        """
        # Get embeddings (uses cache)
        embeddings_a = np.array(self.get_embeddings(anchor_a))
        embeddings_b = np.array(self.get_embeddings(anchor_b))
        
        prompt = contextual_prompt.format(entity=entity_name)
        entity_embedding = np.array(self.get_embedding(prompt))
        
        # Compute centroids
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # Calculate cosine distances (re-calculated for annotations)
        dist_to_legacy = self.cosine_distance(entity_embedding, centroid_a)
        dist_to_strategy = self.cosine_distance(entity_embedding, centroid_b)
        centroid_dist = self.cosine_distance(centroid_a, centroid_b)
        
        # Re-calculate drift score for plot consistency
        axis_vector = centroid_b - centroid_a
        midpoint = (centroid_a + centroid_b) / 2
        entity_relative = entity_embedding - midpoint
        axis_length_sq = np.dot(axis_vector, axis_vector)
        
        if axis_length_sq > 0:
            projection = np.dot(entity_relative, axis_vector) / axis_length_sq
            calculated_drift = float(projection * 2)
        else:
            calculated_drift = 0.0
            
        # Combine all embeddings for PCA
        all_embeddings = np.vstack([
            embeddings_a,
            embeddings_b,
            [centroid_a],
            [centroid_b],
            [entity_embedding],
        ])
        
        # Apply PCA to reduce to 1D
        pca = PCA(n_components=1)
        embeddings_1d = pca.fit_transform(all_embeddings).flatten()
        
        # Split back into groups
        n_a = len(anchor_a)
        n_b = len(anchor_b)
        
        points_a = embeddings_1d[:n_a]
        points_b = embeddings_1d[n_a:n_a + n_b]
        centroid_a_1d = embeddings_1d[n_a + n_b]
        centroid_b_1d = embeddings_1d[n_a + n_b + 1]
        entity_1d = embeddings_1d[n_a + n_b + 2]
        
        # Ensure correct orientation: Legacy (A) should be LEFT, Strategy (B) should be RIGHT
        centroid_a_1d_orig = centroid_a_1d # Store for comparator logic
        centroid_b_1d_orig = centroid_b_1d
        
        if centroid_a_1d > centroid_b_1d:
            points_a = -points_a
            points_b = -points_b
            centroid_a_1d = -centroid_a_1d
            centroid_b_1d = -centroid_b_1d
            entity_1d = -entity_1d
        
        # Normalize axis so 0 = midpoint, Legacy=-1, Strategy=+1
        midpoint_1d = (centroid_a_1d + centroid_b_1d) / 2
        half_range = abs(centroid_b_1d - centroid_a_1d) / 2
        
        if half_range > 0:
            norm_points_a = (points_a - midpoint_1d) / half_range
            norm_points_b = (points_b - midpoint_1d) / half_range
            norm_entity = (entity_1d - midpoint_1d) / half_range
            
            # Process comparators (grouped by type)
            comparator_results = {}
            
            if comparators:
                # Handle both list (backwards compatibility) and dict
                groups = comparators if isinstance(comparators, dict) else {"default": comparators}
                
                for group_name, group_entities in groups.items():
                    group_data = {
                        "norm_positions": [],
                        "drift_scores": [],
                        "names": []
                    }
                    
                    for comp in group_entities:
                        comp_prompt = contextual_prompt.format(entity=comp)
                        comp_embedding = np.array(self.get_embedding(comp_prompt))
                        
                        # Drift Score
                        comp_relative = comp_embedding - midpoint
                        if axis_length_sq > 0:
                            comp_proj = np.dot(comp_relative, axis_vector) / axis_length_sq
                            comp_drift = float(comp_proj * 2)
                        else:
                            comp_drift = 0.0
                        
                        # 1D PCA Projection
                        comp_1d_raw = pca.transform([comp_embedding])[0][0]
                        if centroid_a_1d_orig > centroid_b_1d_orig:
                             comp_1d_raw = -comp_1d_raw
                        
                        comp_norm = (comp_1d_raw - midpoint_1d) / half_range
                        
                        group_data["norm_positions"].append(comp_norm)
                        group_data["drift_scores"].append(comp_drift)
                        group_data["names"].append(comp)
                    
                    comparator_results[group_name] = group_data
                    
        else:
            norm_points_a = np.zeros_like(points_a)
            norm_points_b = np.zeros_like(points_b)
            norm_entity = 0.0
            comparator_results = {}
        
        # Create figure
        fig = go.Figure()
        
        # Draw main axis line (from -1 to +1)
        fig.add_trace(go.Scatter(
            x=[-1, 1],
            y=[0, 0],
            mode="lines",
            line=dict(color=COLORS["axis"], width=4),
            name="Semantic Axis",
            hoverinfo="skip",
        ))
        
        # Plot Legacy anchors (small dots below axis)
        jitter_a = np.random.uniform(-0.15, -0.05, len(anchor_a))
        fig.add_trace(go.Scatter(
            x=norm_points_a,
            y=jitter_a,
            mode="markers+text",
            marker=dict(
                size=8,
                color=COLORS["legacy"],
                opacity=0.7,
                symbol="circle",
            ),
            text=anchor_a,
            textposition="bottom center",
            textfont=dict(size=9, color=COLORS["legacy"], family="Arial"),
            cliponaxis=False,
            name="Legacy Anchors",
            hovertemplate="<b>%{text}</b><br>Position: %{x:.2f}<extra></extra>",
        ))
        
        # Plot Strategy anchors (small dots above axis)
        jitter_b = np.random.uniform(0.05, 0.15, len(anchor_b))
        fig.add_trace(go.Scatter(
            x=norm_points_b,
            y=jitter_b,
            mode="markers+text",
            marker=dict(
                size=8,
                color=COLORS["strategy"],
                opacity=0.7,
                symbol="circle",
            ),
            text=anchor_b,
            textposition="top center",
            textfont=dict(size=9, color=COLORS["strategy"], family="Arial"),
            cliponaxis=False,
            name="Strategy Anchors",
            hovertemplate="<b>%{text}</b><br>Position: %{x:.2f}<extra></extra>",
        ))
        
        # Plot Legacy centroid (large marker at -1)
        fig.add_trace(go.Scatter(
            x=[-1],
            y=[0],
            mode="markers",
            marker=dict(
                size=30,
                color=COLORS["centroid_legacy"],
                symbol="diamond",
                line=dict(color="white", width=3),
            ),
            name="LEGACY",
            hovertemplate="<b>Legacy Centroid</b><br>Position: -1.00<extra></extra>",
        ))
        
        # Plot Strategy centroid (large marker at +1)
        fig.add_trace(go.Scatter(
            x=[1],
            y=[0],
            mode="markers",
            marker=dict(
                size=30,
                color=COLORS["centroid_strategy"],
                symbol="diamond",
                line=dict(color="white", width=3),
            ),
            name="STRATEGY",
            hovertemplate="<b>Strategy Centroid</b><br>Position: +1.00<extra></extra>",
        ))
        
        # Plot Entity (large marker)
        fig.add_trace(go.Scatter(
            x=[norm_entity],
            y=[0],
            mode="markers+text",
            marker=dict(
                size=40,
                color=COLORS["entity"],
                symbol="star",
                line=dict(color="white", width=3),
            ),
            text=[entity_name.upper()],
            textposition="top center",
            textfont=dict(size=14, color=COLORS["entity"], family="Arial Black"),
            name=f"Entity: {entity_name}",
            hovertemplate=(
                f"<b>{entity_name}</b><br>"
                f"Position: %{{x:.3f}}<br>"
                f"Dist to Legacy: {dist_to_legacy:.4f}<br>"
                f"Dist to Strategy: {dist_to_strategy:.4f}<br>"
                f"Drift Score: {calculated_drift:+.4f}"
                "<extra></extra>"
            ),
        ))

        # Plot Comparators (Grouped)
        for group_name, data in comparator_results.items():
            # Determine color and symbol based on group
            if group_name == "legacy":
                color = COLORS["comparator_legacy"]
                symbol = "diamond-open"
            elif group_name == "strategic":
                color = COLORS["comparator_strategic"]
                symbol = "diamond-open"
            else:
                color = "#9370DB" # Default purple
                symbol = "hexagon"
                
            fig.add_trace(go.Scatter(
                x=data["norm_positions"],
                y=[0] * len(data["names"]),
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=color,
                    symbol=symbol,
                    line=dict(color=color, width=2),
                    opacity=0.9,
                ),
                text=data["names"],
                textposition="bottom center",
                textfont=dict(size=10, color=color, family="Arial"),
                name=f"Comparators ({group_name.title()})",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Position: %{x:.3f}<br>"
                    "Drift Score: %{customdata:.4f}"
                    "<extra></extra>"
                ),
                customdata=data["drift_scores"]
            ))
        
        # Add vertical lines at key positions
        fig.add_vline(x=-1, line_width=2, line_color=COLORS["centroid_legacy"], line_dash="dash")
        fig.add_vline(x=1, line_width=2, line_color=COLORS["centroid_strategy"], line_dash="dash")
        fig.add_vline(x=0, line_width=2, line_color="gray", line_dash="dot")  # Neutral line
        
        dim_display = dimension_name.replace("_", " ").title()
        variance = pca.explained_variance_ratio_[0] * 100
        
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Semantic Axis: {dim_display}</b><br>"
                    f"<sub>{entity_name} | PCA 1D ({variance:.1f}% variance explained)</sub>"
                ),
                x=0.5,
                font=dict(size=20, family="Arial"),
            ),
            xaxis=dict(
                title="",
                range=[-1.5, 1.5],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["LEGACY<br>(-1)", "-0.5", "NEUTRAL<br>(0)", "+0.5", "STRATEGY<br>(+1)"],
                tickfont=dict(size=11),
                gridcolor="#E0E0E0",
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                range=[-0.35, 0.35],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            font=dict(family="Arial"),
            height=500,
            width=1100,
            margin=dict(t=100, b=120, l=50, r=50),
            annotations=[
                # Distance labels
                dict(
                    x=norm_entity / 2,
                    y=-0.25,
                    text=f"← Dist: {dist_to_legacy:.3f}",
                    showarrow=False,
                    font=dict(size=11, color=COLORS["legacy"]),
                ),
                dict(
                    x=(norm_entity + 1) / 2,
                    y=-0.25,
                    text=f"Dist: {dist_to_strategy:.3f} →",
                    showarrow=False,
                    font=dict(size=11, color=COLORS["strategy"]),
                ),
                # Drift score
                dict(
                    x=norm_entity,
                    y=0.28,
                    text=f"<b>Drift Score: {calculated_drift:+.4f}</b>",
                    showarrow=False,
                    font=dict(size=14, color=COLORS["entity"]),
                ),
                # Centroid distance
                dict(
                    x=0.5,
                    y=-0.32,
                    text=f"Centroid Separation: {centroid_dist:.4f}",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                ),
            ],
        )
        
        div_id = f"axis_{dimension_name}"
        plot_html = fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id)
        
        # Legacy anchor lists format (kept for redundancy/accessibility below chart)
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
        # Get Plotly JS for offline use
        plotly_js = plotly.offline.get_plotlyjs()

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
    <script type="text/javascript">{plotly_js}</script>
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
            <div class="legend-item"><div class="legend-dot" style="background:#1E90FF"></div> Entity Position</div>
            <div class="legend-item"><div class="legend-dot" style="background:#CD853F"></div> Legacy Peers</div>
            <div class="legend-item"><div class="legend-dot" style="background:#228B22"></div> Strategic Peers</div>
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
