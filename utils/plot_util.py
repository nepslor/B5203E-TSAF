import numpy as np
import plotly.graph_objects as go


def qs_animation_plotly(
    y_te,
    y_hat,
    qs,
    n_rows=50,
    f_name="",
    width=900,
    height=450,
    margin=None,
):
    """
    Animate true values, point forecasts, and quantile confidence intervals.

    Parameters
    ----------
    y_te : array-like, shape (n_samples, T)
        Ground-truth values.
    y_hat : array-like, shape (n_samples, T)
        Point forecasts.
    qs : array-like, shape (n_samples, T, n_quantiles)
        Quantile forecasts ordered from low to high quantile.
    n_rows : int, default=50
        Number of animated samples to display.
    f_name : str, default=""
        Name shown in the figure title.
    width : int, default=900
        Figure width in pixels.
    height : int, default=450
        Figure height in pixels.
    margin : dict | None, default=None
        Plotly margin dict, e.g. {"l": 30, "r": 20, "t": 35, "b": 30}.
    """
    y_te = np.asarray(y_te)
    y_hat = np.asarray(y_hat)
    qs = np.asarray(qs)

    if y_te.ndim != 2 or y_hat.ndim != 2 or qs.ndim != 3:
        raise ValueError("Expected y_te (2D), y_hat (2D), qs (3D).")
    if y_te.shape != y_hat.shape:
        raise ValueError("y_te and y_hat must have the same shape.")
    if y_te.shape[:2] != qs.shape[:2]:
        raise ValueError("qs first two dimensions must match y_te/y_hat.")

    n_samples, horizon = y_te.shape
    n_quantiles = qs.shape[2]

    if n_quantiles < 2:
        raise ValueError("qs must contain at least two quantiles.")

    n_rows = min(n_rows, n_samples)
    t = np.arange(horizon)
    if margin is None:
        margin = {"l": 30, "r": 20, "t": 35, "b": 30}

    # Pair lower/upper quantiles: (0,-1), (1,-2), ...
    n_bands = n_quantiles // 2
    band_pairs = [(k, n_quantiles - 1 - k) for k in range(n_bands)]
    # Wider intervals lighter, inner intervals darker.
    alphas = np.linspace(0.08, 0.35, num=max(n_bands, 1))

    fig = go.Figure(layout=dict(width=width, height=height))

    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_te[0],
            mode="lines",
            line=dict(color="blue", width=2),
            name="y_te",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_hat[0],
            mode="lines",
            line=dict(color="orange", width=2),
            name="y_hat",
        )
    )

    # Add CI bands in red (widest first); filled between lower/upper traces.
    for b, (lo, hi) in enumerate(band_pairs):
        alpha = float(alphas[b])
        rgba = f"rgba(255,0,0,{alpha})"

        fig.add_trace(
            go.Scatter(
                x=t,
                y=qs[0, :, lo],
                mode="lines",
                line=dict(color=rgba, width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=qs[0, :, hi],
                mode="lines",
                line=dict(color=rgba, width=0),
                fill="tonexty",
                fillcolor=rgba,
                name="CI" if b == 0 else None,
                hoverinfo="skip",
                showlegend=(b == 0),
            )
        )

    frames = []
    for i in range(n_rows):
        frame_data = [
            go.Scatter(y=y_te[i]),
            go.Scatter(y=y_hat[i]),
        ]
        for lo, hi in band_pairs:
            frame_data.append(go.Scatter(y=qs[i, :, lo]))
            frame_data.append(go.Scatter(y=qs[i, :, hi]))
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

    fig.frames = frames

    y_min = min(np.min(y_te), np.min(qs)) - 1
    y_max = max(np.max(y_te), np.max(qs)) + 1

    fig.update_layout(
        title=f"Quantile Forecast Animation for {f_name}",
        margin=margin,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ],
    )
    fig.update_yaxes(range=[y_min, y_max], automargin=True)
    fig.update_xaxes(automargin=True)

    return fig
