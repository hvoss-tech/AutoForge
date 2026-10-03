import json
import os
import traceback

import numpy as np


def main():
    # Plotting dependencies only, not the package's own.
    import plotly.graph_objects as go
    from scipy.optimize import curve_fit

    json_file = [f for f in os.listdir("output_grid") if f.endswith(".json")][0]
    # Load the JSON file with arbitrary keys
    with open(f"output_grid/{json_file}", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate means, standard deviations, and store the arrays for each condition.
    # We store as a tuple: (key, mean, std, data array)
    stats = {}
    for run in data:
        param_value = run["param_value"]
        loss = run["loss"]
        stats.setdefault(param_value, []).append(loss)

    # Sort the list of tuples based on the mean (ascending order)
    # stats.sort(key=lambda x: x[1])

    # Extract sorted lists of keys, means, stds, and store the sorted arrays in a dictionary
    sorted_keys = sorted(list(stats.keys()))
    means = [np.mean(stats[key]) for key in sorted_keys]
    stds = [np.std(stats[key]) for key in sorted_keys]
    n_keys = len(sorted_keys)

    # print number of samples for every key:
    for key in sorted_keys:
        print(f"{key}: {len(stats[key])} samples")

    # sprt all by lowest mean value
    # sorted_indices = np.argsort(means)
    # sorted_keys = [sorted_keys[i] for i in sorted_indices]
    # means = [means[i] for i in sorted_indices]
    # stds = [stds[i] for i in sorted_indices]

    # Create x-positions for the bars (one per key)
    x_positions = list(range(n_keys))

    # Create a Plotly bar chart with error bars
    fig = go.Figure()
    for i, key in enumerate(sorted_keys):
        fig.add_trace(
            go.Bar(
                name=key,
                x=[x_positions[i]],
                y=[means[i]],
                error_y=dict(type="data", array=[stds[i]], visible=True),
            )
        )


    # Define an exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c


    # Convert x_positions and means to numpy arrays for curve fitting
    x_data = np.array(x_positions)
    y_data = np.array(means)
    try:
        # Fit the curve
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=(1, 0.1, 0))  # initial guesses

        # Generate smooth x values for the fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = exp_decay(x_fit, *popt)

        # Add the fitted curve to the figure
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name="Exponential Fit",
                line=dict(dash="dash", width=3),
            )
        )
    except Exception:
        traceback.print_exc()

    threshold = 0.05  # 1% improvement
    for i in range(1, len(means)):
        rel_change = (means[i - 1] - means[i]) / means[i - 1]
        print(
            f"{sorted_keys[i - 1]} -> {sorted_keys[i]}: relative change = {rel_change:.4f}"
        )
        if rel_change < threshold:
            print(
                f"Diminishing returns start at x = {sorted_keys[i]} (relative change = {rel_change:.4f})"
            )

    fig.update_layout(
        title="Comparison of Conditions with Mean ± STD (Sorted by Mean)",
        xaxis=dict(
            tickmode="array",
            tickvals=x_positions,
            ticktext=sorted_keys,
            tickangle=45,  # Rotate labels for better readability
        ),
        yaxis_title="Value",
        barmode="group",
    )


    fig.update_layout(
        autosize=False,
        width=1920,
        height=800,
    )
    # save image
    fig.write_image("out.png")


if __name__ == "__main__":
    main()
