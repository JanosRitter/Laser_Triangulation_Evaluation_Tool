from __future__ import annotations

from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def save_interactive_point_cloud_html(
    triangulated_points: np.ndarray,
    output_path: str | Path,
    title: str = "Interactive Point Cloud",
    annotate_frame_idx: bool = False,
) -> Path:
    points = np.asarray(triangulated_points, dtype=float)

    if points.ndim != 2:
        raise ValueError("triangulated_points muss 2D sein.")

    if points.shape[1] >= 5:
        frame_idx = points[:, 0].astype(int)
        x = points[:, 2]
        y = points[:, 3]
        z = points[:, 4]
    elif points.shape[1] == 3:
        frame_idx = np.arange(len(points))
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    else:
        raise ValueError(f"Unerwartetes Punkteformat: {points.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = [str(i) for i in frame_idx] if annotate_frame_idx else None

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text" if annotate_frame_idx else "markers",
                text=text,
                marker=dict(size=3),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(f"💾 Interaktive Punktwolke gespeichert: {output_path}")

    return output_path