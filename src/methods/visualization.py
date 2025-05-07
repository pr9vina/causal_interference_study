import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# constants for graphs
METHOD_COL = "Feature Names"
# order for methods for plots
METHOD_ORDER = [
    'Naive',
    'community',
    'degree_centrality, betweenness_centrality',
    'degree_centrality, betweenness_centrality, community',
    'Node2Vec',
    'treated_neighbors_count',
    'treated_neighbors_exposure'
]
# distance / location of methods in plots
OFFSET = {
    'barabasi_albert_graph': -0.2,
    'watts_strogatz_graph': 0.0,
    'erdos_renyi_graph': 0.2
}
COLORS = {
    'barabasi_albert_graph': '#66c2a5',
    'watts_strogatz_graph': '#fc8d62',
    'erdos_renyi_graph': '#8da0cb'
}
CORRUPT_SETTINGS_STYLE = {
        "Full": {"marker": "o", "alpha": 1.0},
        "Corrupt": {"marker": "s", "alpha": 0.6}
    }
METHOD_LABELS = {
    'Naive': 'Naive',
    'community': 'Сommunity',
    'degree_centrality, betweenness_centrality': 'Centrality',
    'degree_centrality, betweenness_centrality, community': 'Centrality and community',
    'Node2Vec': 'Node2Vec',
    'treated_neighbors_count': 'Treated count',
    'treated_neighbors_exposure': 'Treated exposure'
}
LABEL_NAMES = {
        "barabasi_albert_graph": "Barabasi–Albert",
        "watts_strogatz_graph": "Watts–Strogatz",
        "erdos_renyi_graph": "Erdos–Renyi"
}


class ResultsPlotter:
    def __init__(
        self,
        estimated_effect: np.array,
        true_effect: float,
        network_type: str,
        assignment_type: str,
        neighbour_influence: float,
        additional_feature_names: list[str],
        bias_statistics: list[float],
        pvalues: list[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ):
        """Initialize the ResultsPlotter with estimation and simulation parameters.

        Args:
            estimated_effect (np.array): array of estimated treatment effects
            true_effect (float): true treatment effect used in simulation
            network_type (str): type of network structure)
            assignment_type (str): treatmet assignment method
            neighbour_influence (float): neigbour influence used in simulation
            additional_feature_names (list[str]): covariates used in adjustment methods
            bias_statistics (list[float]): calculated bias statistics from simulations
            pvalues (list[float]): calculated pvalues in simulations
            n_bootstrap (int, optional): number of bootrstraps for CI. Defaults to 10000.
            confidence (float, optional): confidence level for bootstrap and FPR calculation. Defaults to 0.95.
        """        
        self.estimated_effect = estimated_effect
        self.true_effect = true_effect
        self.network_type = network_type
        self.assignment_type = assignment_type
        self.neighbour_influence = neighbour_influence
        self.additional_feature_names = additional_feature_names
        self.bias_statistics = bias_statistics
        self.pvalues = pvalues
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence

    def _plot_histogram(self, ax, data, color, title, line_color='black'):
        ax.hist(data, bins=30, alpha=0.7, color=color)
        ax.axvline(np.mean(data), color=line_color, linestyle='--', linewidth=2)
        ax.set_title(title)

    def _calculate_percentage_difference(self, mean_estimated, mean_true):
        return np.abs((mean_estimated - mean_true) / mean_true) * 100

    def _save_plot(self, fig):
        graph_name = f"{self.network_type}_{self.assignment_type}_{self.neighbour_influence}_{self.true_effect}.png"
        plot_filename = f"visualization_results/{graph_name}"
        fig.savefig(plot_filename)
        plt.close(fig)
        logging.info(f"Plot saved to {plot_filename}")

    def _generate_plot_name(self):
        return f"Network: {self.network_type}, Assignment: {self.assignment_type}, Influence: {self.neighbour_influence}, Treatment: {self.true_effect}, Perc Diff: {self.perc_diff}"

    def create_effect_estimation_plots(self):
        """Create distribution of true and estimated effects"""        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self._plot_histogram(ax1, self.estimated_effect, 'blue', f'Estimated effects: {np.round(self.mean_estimated, 3)}')
        self._plot_histogram(ax2, self.true_effect, 'green', f'True effects: {np.round(self.mean_true, 3)}')
        logging.info(f"Percentage difference between true and estimated effect: {np.round(self.perc_diff, 3)}")
        graph_name = self._generate_plot_name()
        fig.suptitle(graph_name, fontsize=8, fontweight='bold')
        self._save_plot(fig)

    def _bootstrap_ci(self) -> (float, float):
        """Calculate CI of bias statistics"""       
        boot_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(self.bias_statistics, size=len(self.bias_statistics), replace=True)
            boot_means.append(np.mean(sample))
        lower = np.percentile(boot_means, (1 - self.confidence) / 2 * 100)
        upper = np.percentile(boot_means, (1 + self.confidence) / 2 * 100)
        return lower, upper

    def calculate_results(self):
        """Calculate aggregated results from simulations"""        
        self.mean_estimated = np.mean(self.estimated_effect)
        self.mean_true = np.mean(self.true_effect)
        self.perc_diff = self._calculate_percentage_difference(self.mean_estimated, self.mean_true)
        self.mean_bias = np.mean(self.bias_statistics)
        self.lower_bound, self.upper_bound = self._bootstrap_ci()
        self.FPR = sum(p < (1-self.confidence) for p in self.pvalues) / len(self.pvalues)

    def simplify_feature_names(self, features):
        """Supporting function for simlification of names of adjustments"""   
        if features is None:
            return "Naive"
        if isinstance(features, str) and "emb_0" in features:
            return "Node2Vec"
        if isinstance(features, (list, tuple)):
            if "emb_0" in features:
                return "Node2Vec"
            if "treated_neighbors_exposure" in features:
                return "treated_neighbors_exposure"
            if "treated_neighbors_count" in features:
                return "treated_neighbors_count"
            if "degree_centrality" in features and "community" in features:
                return "degree_centrality, betweenness_centrality, community"
            if "degree_centrality" in features:
                return "degree_centrality, betweenness_centrality"
            if "community" in features:
                return "community"   
        return features

    def create_results(self) -> pd.DataFrame:
        """Create results table for simulations."""   
        results_df = pd.DataFrame([{
            "Network": self.network_type,
            "Assignment": self.assignment_type,
            "Influence": self.neighbour_influence,
            "Treatment": self.true_effect,
            "Feature Names": self.additional_feature_names,
            "Mean Estimated": round(self.mean_estimated, 3),
            "Mean True": round(self.mean_true, 3),
            "Perc Diff Overall (%)": round(self.perc_diff, 3),
            "Mean Bias": round(self.mean_bias, 3),
            "Bias Lower Bound (95%)": self.lower_bound,
            "Bias Upper Bound (95%)": self.upper_bound,
            "FPR": round(self.FPR, 3)
        }])
        results_df["Feature Names"] = results_df["Feature Names"].apply(self.simplify_feature_names)
        table_name = f"markdown_results/new/{self.network_type}_{self.assignment_type}_{self.neighbour_influence}_{self.true_effect}.pkl"
        results_df.to_pickle(table_name)
        return results_df


def plot_result_graphs(
    all_results: pd.DataFrame,
    influence: float,
    method_col: str = METHOD_COL,
    method_order: list[str] = METHOD_ORDER,
    offset: dict[str, float] = OFFSET,
    colors: dict[str, str] = COLORS,
    label_names: list[str] = LABEL_NAMES
):
    """Plot resulting graphs by influence. All network structures are present.
    NB! This representation isn't used in the final paprt

    Args:
        all_results (pd.DataFrame): table with resulted simulations
        influence (float): neighbour influence 
        method_order (list[str], optional): order of methods in graphs. Defaults to METHOD_ORDER.
        offset (dict[str, float], optional): offset of methods in graphs. Defaults to OFFSET.
        colors (dict[str, str], optional): colors of network structure. Defaults to COLORS.
        label_names (list[str], optional): labels for network structures. Defaults to LABEL_NAMES.
    """

    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(method_order))
    for network in all_results["Network"].unique():
        df = all_results[(all_results["Network"] == network)]

        x_vals = [method_order.index(m) + offset[network] for m in df[method_col]]
        mean_bias = df["Mean Bias"].values
        lower_err = (mean_bias - df["Bias Lower Bound (95%)"]).values
        upper_err = (df["Bias Upper Bound (95%)"] - mean_bias).values

        plt.errorbar(
            x=x_vals,
            y=mean_bias,
            yerr=[lower_err, upper_err],
            fmt='o',
            capsize=5,
            label=label_names.get(network, network),
            color=colors[network]
        )
        for x, y in zip(x_vals, mean_bias):
            plt.text(x, y, f"{y:.2f}", color="black", ha='center', va='center', fontsize=8)

    short_labels = [METHOD_LABELS[m] for m in method_order]
    plt.xticks(ticks=x_pos, labels=short_labels, rotation=30, ha="right")
    plt.ylabel("Mean Bias")
    plt.xlabel("Adjustment Method")
    plt.legend(title="Network")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fpr_by_method(
    df: pd.DataFrame,
    influence: float,
    method_col: str = METHOD_COL,
    method_order: list[str] = METHOD_ORDER,
    label_names: dict[str, str] = LABEL_NAMES
):
    """Plot FPR by method

    Args:
        df (pd.DataFrame): results table
        influence (float): neighbour influence
        method_col (str, optional): name of method column. Defaults to METHOD_COL.
        method_order (list[str], optional): order of methods. Defaults to METHOD_ORDER.
        label_names (dict[str, str], optional): labels for network structures. Defaults to LABEL_NAMES.
    """    
    df = df.copy()
    df[method_col] = pd.Categorical(df[method_col], categories=method_order, ordered=True)
    df = df.sort_values([method_col, "Network"])
    df["Network"] = df["Network"].map(lambda n: label_names.get(n, n))
    short_labels = [METHOD_LABELS[m] for m in method_order]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x=method_col,
        y="FPR",
        hue="Network",
        order=method_order,
        palette="Set2",
        edgecolor="black",
        errwidth=0,
        ci=None
    )

    plt.axhline(0.05, color='red', linestyle='--', linewidth=1.2, label='FPR Threshold (0.05)')
    plt.ylabel("False Positive Rate (FPR)")
    plt.xlabel("Adjustment Method")
    plt.title(f"False Positive Rate by Method (Influence = {influence})")

    plt.xticks(
        ticks=np.arange(len(method_order)),
        labels=short_labels,
        rotation=30,
        ha="right"
    )
    plt.legend(title="Network")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_result_graphs_multiple_influence(
    all_results,
    influence_column="Influence",
    method_order=METHOD_ORDER,
    offset=OFFSET,
    colors=COLORS,
    label_names=LABEL_NAMES
):
    influence_levels = sorted(all_results[influence_column].unique())
    n_levels = len(influence_levels)

    fig, axes = plt.subplots(1, n_levels, figsize=(8 * n_levels, 6), sharey=False)

    if n_levels == 1:
        axes = [axes]

    for idx, influence in enumerate(influence_levels):
        ax = axes[idx]
        subset = all_results[all_results[influence_column] == influence]
        x_pos = np.arange(len(method_order))

        for network in subset["Network"].unique():
            df = subset[subset["Network"] == network]
            x_vals = [method_order.index(m) + offset[network] for m in df["Feature Names"]]
            mean_bias = df["Mean Bias"].values
            lower_err = (mean_bias - df["Bias Lower Bound (95%)"]).values
            upper_err = (df["Bias Upper Bound (95%)"] - mean_bias).values

            ax.errorbar(
                x=x_vals,
                y=mean_bias,
                yerr=[lower_err, upper_err],
                fmt='o',
                capsize=5,
                label=label_names.get(network, network),
                color=colors[network]
            )

            for x, y in zip(x_vals, mean_bias):
                ax.text(x, y, f"{y:.2f}", color="black", ha='center', va='center', fontsize=9)

        short_labels = [label_names[m] for m in method_order]
        ax.set_xticks(x_pos)
        ax.set_ylim(bottom=0)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=11)
        ax.set_title(f"{influence_column} = {influence}", fontsize=13)
        ax.set_xlabel("Adjustment Method", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean Bias", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, title="Network", loc="lower center", ncol=3, fontsize=11, title_fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_bias_with_corruption_overlay(
    df_full: pd.DataFrame,
    df_corrupt: pd.DataFrame,
    influence: float,
    method_order: list[str] = METHOD_ORDER,
    network_colors: dict[str, str] = COLORS,
    setting_styles: dict[str, str] = CORRUPT_SETTINGS_STYLE
):
    """Plot bias for fully vs partial observed network structures

    Args:
        df_full (pd.DataFrame): results for full oberved networks
        df_corrupt (pd.DataFrame): result for partially observed networks
        influence (float): neighbour influence
        method_order (list[str], optional): order of methods. Defaults to METHOD_ORDER.
        network_colors (dict[str, str], optional): colors for network structures. Defaults to COLORS.
        setting_styles (dict[str, str], optional): styles for partial observed networks. Defaults to CORRUPT_SETTINGS_STYLE.
    """

    df_full = df_full[np.isclose(df_full["Influence"], influence)].copy()
    df_full["Setting"] = "Full"
    df_corrupt = df_corrupt[np.isclose(df_corrupt["Influence"], influence)].copy()
    df_corrupt["Setting"] = "Corrupt"
    df = pd.concat([df_full, df_corrupt])
    df["Feature Names"] = pd.Categorical(df["Feature Names"], method_order, ordered=True)
    network_list = list(network_colors.keys())
    setting_list = ["Full", "Corrupt"]
    total_groups = len(network_list) * len(setting_list)
    group_offsets = {
        (net, setting): (-0.3 + 0.6 * (i / (total_groups - 1)))
        for i, (net, setting) in enumerate(
            [(n, s) for n in network_list for s in setting_list]
        )
    }

    plt.figure(figsize=(14, 6))

    for i, method in enumerate(method_order):
        for net in network_list:
            for setting in setting_list:
                row = df[(df["Feature Names"] == method) &
                         (df["Network"] == net) &
                         (df["Setting"] == setting)]

                if not row.empty:
                    y = row["Mean Bias"].values[0]
                    yerr = [[y - row["Bias Lower Bound (95%)"].values[0]],
                            [row["Bias Upper Bound (95%)"].values[0] - y]]
                    x = i + group_offsets[(net, setting)]
                    plt.errorbar(
                        x=x, y=y,
                        yerr=np.abs(yerr),
                        fmt=setting_styles[setting]["marker"],
                        color=network_colors[net],
                        capsize=4,
                        markersize=6,
                        alpha=setting_styles[setting]["alpha"],
                        label=f"{net} ({setting})" if i == 0 else ""
                    )
                    plt.text(x, y, f"{y:.2f}", color="black", fontsize=8,
                             ha='center', va='center', zorder=10)

    plt.axhline(y=0, linestyle="--", color="gray", alpha=0.6)
    plt.xticks(ticks=np.arange(len(method_order)), labels=method_order, rotation=30, ha='right')
    plt.ylabel("Mean Bias")
    plt.xlabel("Adjustment Method")
    plt.title(f"Mean Bias with 95% CI for Full and Corrupt Networks (Influence = {influence})")
    plt.legend(title="Network + Setting", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bias_full_corruption(
    df_full,
    df_corrupt,
    method_order=METHOD_LABELS.keys(),
    network_colors=COLORS,
    setting_styles=CORRUPT_SETTINGS_STYLE,
    label_names=LABEL_NAMES,
    influence_column="Influence",
    method_labels=None
):
    """Plot bias

    Args:
        df_full (_type_): _description_
        df_corrupt (_type_): _description_
        method_order (_type_, optional): _description_. Defaults to METHOD_LABELS.keys().
        network_colors (_type_, optional): _description_. Defaults to COLORS.
        setting_styles (_type_, optional): _description_. Defaults to CORRUPT_SETTINGS_STYLE.
        label_names (_type_, optional): _description_. Defaults to LABEL_NAMES.
        influence_column (str, optional): _description_. Defaults to "Influence".
        method_labels (_type_, optional): _description_. Defaults to None.
    """    
    df_full = df_full.copy()
    df_full["Setting"] = "Full"
    df_corrupt = df_corrupt.copy()
    df_corrupt["Setting"] = "Corrupt"
    df = pd.concat([df_full, df_corrupt])
    df["Feature Names"] = pd.Categorical(df["Feature Names"], method_order, ordered=True)

    influence_levels = sorted(df[influence_column].unique())
    n_levels = len(influence_levels)

    fig, axes = plt.subplots(1, n_levels, figsize=(7 * n_levels, 6), sharey=False)
    if n_levels == 1:
        axes = [axes]

    for idx, influence in enumerate(influence_levels):
        ax = axes[idx]
        subset = df[np.isclose(df[influence_column], influence)]

        for i, method in enumerate(method_order):
            for net in subset["Network"].unique():
                for setting in ["Full", "Corrupt"]:
                    row = subset[
                        (subset["Feature Names"] == method) &
                        (subset["Network"] == net) &
                        (subset["Setting"] == setting)
                    ]
                    if not row.empty:
                        y = row["Mean Bias"].values[0]
                        lower = row["Bias Lower Bound (95%)"].values[0]
                        upper = row["Bias Upper Bound (95%)"].values[0]
                        yerr = [[y - lower], [upper - y]]
                        x = i + (-0.1 if setting == "Full" else 0.25)

                        base_color = network_colors.get(net, "gray")
                        color = base_color
                        if setting == "Corrupt":
                            color = mcolors.to_rgba(base_color, alpha=setting_styles[setting]["alpha"])

                        ax.errorbar(
                            x=x,
                            y=y,
                            yerr=yerr,
                            fmt=setting_styles[setting]["marker"],
                            color=color,
                            capsize=4,
                            markersize=6,
                            alpha=setting_styles[setting]["alpha"],
                        )
                        ax.text(
                            x, y + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # смещение вверх
                            f"{y:.2f}",
                            color="black",
                            fontsize=8,
                            ha='center',
                            va='bottom',  # под текстом
                            zorder=10
                        )


        xtick_labels = [method_labels.get(m, m) if method_labels else m for m in method_order]
        ax.set_ylim(bottom=-1)
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=11)
        ax.set_title(f"{influence_column} = {influence}", fontsize=13)
        ax.set_xlabel("Adjustment Method", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean Bias", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    

    networks_in_plot = subset["Network"].unique()
    combined_legend = [
    Line2D(
        [0], [0],
        marker=setting_styles[setting]["marker"],
        color=mcolors.to_rgba(network_colors[net], alpha=setting_styles[setting]["alpha"]),
        label=f"{label_names.get(net, net)} ({setting})",
        linestyle='None',
        markersize=6
    )
    for net in networks_in_plot
    for setting in setting_styles
]

    ax.legend(
        handles=combined_legend,
        title="Network settings",
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=9,
        title_fontsize=10
    )
    plt.show()


def plot_bias_full_corruption(
    df_full,
    df_corrupt,
    method_order=METHOD_LABELS.keys(),
    network_colors=COLORS,
    setting_styles=CORRUPT_SETTINGS_STYLE,
    label_names=LABEL_NAMES,
    influence_column="Influence",
    method_labels=None
):
    df_full = df_full.copy()
    df_full["Setting"] = "Full"
    df_corrupt = df_corrupt.copy()
    df_corrupt["Setting"] = "Corrupt"
    df = pd.concat([df_full, df_corrupt])
    df["Feature Names"] = pd.Categorical(df["Feature Names"], method_order, ordered=True)

    influence_levels = sorted(df[influence_column].unique())
    n_levels = len(influence_levels)

    fig, axes = plt.subplots(1, n_levels, figsize=(7 * n_levels, 6), sharey=False)
    if n_levels == 1:
        axes = [axes]

    for idx, influence in enumerate(influence_levels):
        ax = axes[idx]
        subset = df[np.isclose(df[influence_column], influence)]

        for i, method in enumerate(method_order):
            for net in subset["Network"].unique():
                for setting in ["Full", "Corrupt"]:
                    row = subset[
                        (subset["Feature Names"] == method) &
                        (subset["Network"] == net) &
                        (subset["Setting"] == setting)
                    ]
                    if not row.empty:
                        y = row["Mean Bias"].values[0]
                        lower = row["Bias Lower Bound (95%)"].values[0]
                        upper = row["Bias Upper Bound (95%)"].values[0]
                        yerr = [[y - lower], [upper - y]]
                        x = i + (-0.1 if setting == "Full" else 0.25)

                        base_color = network_colors.get(net, "gray")
                        color = base_color
                        if setting == "Corrupt":
                            color = mcolors.to_rgba(base_color, alpha=setting_styles[setting]["alpha"])

                        ax.errorbar(
                            x=x,
                            y=y,
                            yerr=yerr,
                            fmt=setting_styles[setting]["marker"],
                            color=color,
                            capsize=4,
                            markersize=6,
                            alpha=setting_styles[setting]["alpha"],
                        )
                        ax.text(
                            x, y + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # смещение вверх
                            f"{y:.2f}",
                            color="black",
                            fontsize=8,
                            ha='center',
                            va='bottom',  # под текстом
                            zorder=10
                        )


        xtick_labels = [method_labels.get(m, m) if method_labels else m for m in method_order]
        ax.set_ylim(bottom=-1)
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=11)
        ax.set_title(f"{influence_column} = {influence}", fontsize=13)
        ax.set_xlabel("Adjustment Method", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean Bias", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    

    networks_in_plot = subset["Network"].unique()
    combined_legend = [
    Line2D(
        [0], [0],
        marker=setting_styles[setting]["marker"],
        color=mcolors.to_rgba(network_colors[net], alpha=setting_styles[setting]["alpha"]),
        label=f"{label_names.get(net, net)} ({setting})",
        linestyle='None',
        markersize=6
    )
    for net in networks_in_plot
    for setting in setting_styles
]

    ax.legend(
        handles=combined_legend,
        title="Network settings",
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=9,
        title_fontsize=10
    )
    plt.show()