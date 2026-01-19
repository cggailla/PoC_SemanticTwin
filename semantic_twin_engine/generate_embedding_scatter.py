"""
Embedding Axis Plot Generator.

Creates a 1D visualization of the semantic axis for a dimension,
showing the entity position between Legacy and Strategy centroids.
"""

import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env file for OPENAI_API_KEY

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import yaml

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.base_probe import BaseProbe

# Color palette
COLORS = {
    "legacy": "#8B4513",       # Saddle brown
    "legacy_light": "rgba(139, 69, 19, 0.3)",
    "strategy": "#2E8B57",     # Sea green
    "strategy_light": "rgba(46, 139, 87, 0.3)",
    "entity": "#1E90FF",       # Dodger blue
    "centroid_legacy": "#CD853F",   # Peru (darker legacy)
    "centroid_strategy": "#228B22",  # Forest green (darker strategy)
    "axis": "#333333",
}


class EmbeddingAxisGenerator(BaseProbe):
    """Generates 1D axis plots of semantic positioning.
    
    Uses PCA to reduce to 1D and shows entity position on the axis
    between Legacy and Strategy centroids.
    """
    
    @property
    def name(self) -> str:
        return "embedding_axis"
    
    def run(self, context):
        """Not used - this class is for visualization only."""
        pass
    
    def generate_dimension_axis(
        self,
        dimension_name: str,
        anchor_a: list[str],
        anchor_b: list[str],
        contextual_prompt: str,
        entity_name: str,
    ) -> go.Figure:
        """Generate 1D axis plot for a single dimension.
        
        Args:
            dimension_name: Name of the dimension.
            anchor_a: List of legacy anchor terms.
            anchor_b: List of strategy anchor terms.
            contextual_prompt: The contextual prompt template.
            entity_name: Name of the entity to analyze.
        
        Returns:
            Plotly Figure with the 1D axis plot.
        """
        print(f"\n📊 Generating axis plot for dimension: {dimension_name}")
        print(f"   Legacy anchors: {len(anchor_a)} terms")
        print(f"   Strategy anchors: {len(anchor_b)} terms")
        
        # Get all embeddings
        print("   Fetching embeddings for legacy anchors...")
        embeddings_a = np.array(self.get_embeddings(anchor_a))
        
        print("   Fetching embeddings for strategy anchors...")
        embeddings_b = np.array(self.get_embeddings(anchor_b))
        
        # Get entity embedding
        prompt = contextual_prompt.format(entity=entity_name)
        print(f"   Fetching entity embedding...")
        print(f"   Prompt: \"{prompt[:60]}...\"")
        entity_embedding = np.array(self.get_embeddings([prompt])[0])
        
        # Compute centroids
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # Calculate cosine distances (in original high-dim space)
        dist_to_legacy = self.cosine_distance(entity_embedding, centroid_a)
        dist_to_strategy = self.cosine_distance(entity_embedding, centroid_b)
        centroid_dist = self.cosine_distance(centroid_a, centroid_b)
        
        # Calculate drift score using projection (centered at 0)
        # Same formula as main engine: project onto A→B axis relative to midpoint
        axis_vector = centroid_b - centroid_a
        midpoint = (centroid_a + centroid_b) / 2
        entity_relative = entity_embedding - midpoint
        axis_length_sq = np.dot(axis_vector, axis_vector)
        
        if axis_length_sq > 0:
            projection = np.dot(entity_relative, axis_vector) / axis_length_sq
            drift_score = float(projection * 2)  # Scale to -1 to +1
        else:
            drift_score = 0.0
        
        print(f"   Distance to Legacy: {dist_to_legacy:.4f}")
        print(f"   Distance to Strategy: {dist_to_strategy:.4f}")
        print(f"   Distance between centroids: {centroid_dist:.4f}")
        print(f"   Drift Score: {drift_score:+.4f}")
        
        # Combine all embeddings for PCA
        all_embeddings = np.vstack([
            embeddings_a,
            embeddings_b,
            [centroid_a],
            [centroid_b],
            [entity_embedding],
        ])
        
        # Apply PCA to reduce to 1D
        print("   Applying 1D PCA projection...")
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
        # PCA might flip the axis, so we check and flip if needed
        if centroid_a_1d > centroid_b_1d:
            # Flip everything
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
        else:
            norm_points_a = np.zeros_like(points_a)
            norm_points_b = np.zeros_like(points_b)
            norm_entity = 0.0
        
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
            textfont=dict(size=7, color=COLORS["legacy"]),
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
            textfont=dict(size=7, color=COLORS["strategy"]),
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
                f"Drift Score: {drift_score:+.4f}"
                "<extra></extra>"
            ),
        ))
        
        # Add vertical lines at key positions
        fig.add_vline(x=-1, line_width=2, line_color=COLORS["centroid_legacy"], line_dash="dash")
        fig.add_vline(x=1, line_width=2, line_color=COLORS["centroid_strategy"], line_dash="dash")
        fig.add_vline(x=0, line_width=2, line_color="gray", line_dash="dot")  # Neutral line
        
        # Add distance annotations
        variance_explained = pca.explained_variance_ratio_[0] * 100
        
        # Update layout
        dim_display = dimension_name.replace("_", " ").title()
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Semantic Axis: {dim_display}</b><br>"
                    f"<sub>{entity_name} | PCA 1D ({variance_explained:.1f}% variance explained)</sub>"
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
                    text=f"<b>Drift Score: {drift_score:+.4f}</b>",
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
        
        return fig


def load_dimension_config(settings_path: Path, dimension_name: str) -> dict:
    """Load dimension configuration from settings.yaml.
    
    Args:
        settings_path: Path to settings.yaml.
        dimension_name: Name of the dimension to load.
    
    Returns:
        Dictionary with anchor_a, anchor_b, and contextual_prompt.
    """
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    
    dimensions = settings["probes"]["vector_probe"]["params"]["dimensions"]
    
    if dimension_name not in dimensions:
        available = list(dimensions.keys())
        raise ValueError(f"Dimension '{dimension_name}' not found. Available: {available}")
    
    dim_config = dimensions[dimension_name]
    return {
        "anchor_a": dim_config["anchor_a"],
        "anchor_b": dim_config["anchor_b"],
        "contextual_prompt": dim_config["contextual_prompt"],
    }


def main(dimension_name: str = "product_physics") -> int:
    """Generate embedding axis plot for a dimension.
    
    Args:
        dimension_name: The dimension to visualize.
    
    Returns:
        Exit code.
    """
    base_path = Path(__file__).parent
    settings_path = base_path / "config" / "settings.yaml"
    output_dir = base_path / "output" / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load settings
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    
    entity_name = settings["entity"]["name"]
    
    print(f"📍 Entity: {entity_name}")
    print(f"📍 Dimension: {dimension_name}")
    
    # Load dimension config
    try:
        dim_config = load_dimension_config(settings_path, dimension_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Generate visualization
    generator = EmbeddingAxisGenerator()
    
    fig = generator.generate_dimension_axis(
        dimension_name=dimension_name,
        anchor_a=dim_config["anchor_a"],
        anchor_b=dim_config["anchor_b"],
        contextual_prompt=dim_config["contextual_prompt"],
        entity_name=entity_name,
    )
    
    # Save
    output_path = output_dir / f"{dimension_name}_embedding_axis.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\nOpen in browser: {output_path.absolute()}")
    webbrowser.open(str(output_path.absolute()))
    
    return 0


if __name__ == "__main__":
    # Default to product_physics, or take argument
    dim = sys.argv[1] if len(sys.argv) > 1 else "product_physics"
    sys.exit(main(dim))
