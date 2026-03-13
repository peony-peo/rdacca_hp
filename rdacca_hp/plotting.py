# plotting.py (优化版本)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import to_rgba
import warnings
from typing import Union, List, Optional

try:
    import seaborn as sns

    _has_seaborn = True
except ImportError:
    _has_seaborn = False
    warnings.warn("seaborn not available, using matplotlib for plotting")

from .core import RdaccaHpResult


def plot_rdaccahp(result: RdaccaHpResult,
                  plot_type: str = "bar",
                  plot_perc: bool = False,
                  color: Optional[Union[str, List[str]]] = None,
                  figsize: tuple = (10, 6),
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  fontsize: int = 12,
                  **kwargs) -> plt.Figure:
    """
    Plot for rdacca_hp results

    Parameters
    ----------
    result : RdaccaHpResult
        Result object from rdacca_hp analysis
    plot_type : str
        Type of plot: "bar" for bar plot, "venn" for Venn diagram
    plot_perc : bool
        For bar plots, whether to show percentages
    color : str or list, optional
        Colors for the plot
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    fontsize : int
        Font size for labels
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if not isinstance(result, RdaccaHpResult):
        raise ValueError("result should be a RdaccaHpResult object")

    if plot_type == "bar":
        return _plot_bar(result, plot_perc, color, figsize, title, xlabel, ylabel, fontsize, **kwargs)
    elif plot_type == "venn":
        return _plot_venn(result, color, figsize, title, fontsize, **kwargs)
    else:
        raise ValueError("plot_type must be 'bar' or 'venn'")


def _plot_bar(result: RdaccaHpResult,
              plot_perc: bool = False,
              color: Optional[Union[str, List[str]]] = None,
              figsize: tuple = (10, 6),
              title: Optional[str] = None,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              fontsize: int = 12,
              **kwargs) -> plt.Figure:
    """
    Create bar plot for hierarchical partitioning results
    """
    hier_part = result.hier_part
    n_vars = len(hier_part)

    # Prepare data for plotting
    if plot_perc:
        values = hier_part["I.perc(%)"].values
        default_ylabel = "% Individual effect to Rsquare (%I)"
    else:
        values = hier_part["Individual"].values
        default_ylabel = "Individual effect"

    variable_names = hier_part.index.tolist()

    # Sort by values (descending)
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = values[sorted_indices]
    sorted_names = [variable_names[i] for i in sorted_indices]

    # Set up colors
    if color is None:
        if _has_seaborn:
            colors = sns.color_palette("husl", n_vars)
        else:
            # Use matplotlib default colors
            colors = [f'C{i}' for i in range(n_vars)]
    elif isinstance(color, str):
        colors = [color] * n_vars
    else:
        colors = color
        if len(colors) < n_vars:
            colors = colors * (n_vars // len(colors) + 1)
        colors = colors[:n_vars]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    bars = ax.bar(range(n_vars), sorted_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(sorted_values),
                f'{value:.2f}' + ('%' if plot_perc else ''),
                ha='center', va='bottom', fontsize=fontsize - 2)

    # Customize plot
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel(ylabel or default_ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel or "Variables", fontsize=fontsize)

    # Set title
    method, rtype = result.method_type
    if title is None:
        title = f"{method} - Hierarchical Partitioning ({rtype})"
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    return fig


def _plot_venn(result: RdaccaHpResult,
               color: Optional[Union[str, List[str]]] = None,
               figsize: tuple = (8, 8),
               title: Optional[str] = None,
               fontsize: int = 12,
               **kwargs) -> plt.Figure:
    """
    Create Venn diagram for variation partitioning results
    """
    if result.var_part is None:
        raise ValueError("Variation partitioning results not available. Run rdacca_hp with var_part=True.")

    hier_part = result.hier_part
    var_part = result.var_part
    n_vars = len(hier_part)

    if n_vars not in [2, 3, 4]:
        raise ValueError(f"Venn diagram supports only 2-4 variables, got {n_vars}")

    # Extract values for Venn diagram
    # The order depends on the number of variables
    if n_vars == 2:
        # Order: A, B, AB
        values = [
            var_part.iloc[0]["Fractions"],  # Unique to A
            var_part.iloc[1]["Fractions"],  # Unique to B
            var_part.iloc[2]["Fractions"]  # Common to A and B
        ]
    elif n_vars == 3:
        # Order: A, B, C, AB, AC, BC, ABC
        values = [
            var_part.iloc[0]["Fractions"],  # Unique to A
            var_part.iloc[1]["Fractions"],  # Unique to B
            var_part.iloc[2]["Fractions"],  # Unique to C
            var_part.iloc[3]["Fractions"],  # Common to A and B
            var_part.iloc[5]["Fractions"],  # Common to A and C (note: R uses different order)
            var_part.iloc[4]["Fractions"],  # Common to B and C
            var_part.iloc[6]["Fractions"]  # Common to A, B, and C
        ]
    else:  # n_vars == 4
        # Order for 4-set Venn: more complex, we'll implement a simplified version
        values = _get_4set_venn_values(var_part)

    # Set up colors
    if color is None:
        if _has_seaborn:
            colors = sns.color_palette("husl", n_vars)
        else:
            colors = [f'C{i}' for i in range(n_vars)]
    elif isinstance(color, str):
        colors = [color] * n_vars
    else:
        colors = color
        if len(colors) < n_vars:
            colors = colors * (n_vars // len(colors) + 1)
        colors = colors[:n_vars]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw Venn diagram based on number of variables
    if n_vars == 2:
        _draw_venn2(ax, values, hier_part.index.tolist(), colors, fontsize)
    elif n_vars == 3:
        _draw_venn3(ax, values, hier_part.index.tolist(), colors, fontsize)
    else:  # n_vars == 4
        _draw_venn4(ax, values, hier_part.index.tolist(), colors, fontsize)

    # Set title
    method, rtype = result.method_type
    if title is None:
        title = f"{method} - Variation Partitioning ({rtype})"
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=20)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    return fig


def _get_4set_venn_values(var_part: pd.DataFrame) -> List[float]:
    """
    Extract values for 4-set Venn diagram in the correct order
    """
    # For 4 variables, the order in var_part is:
    # 0: Unique to A, 1: Unique to B, 2: Unique to C, 3: Unique to D,
    # 4: Common to A and B, 5: Common to A and C, 6: Common to A and D,
    # 7: Common to B and C, 8: Common to B and D, 9: Common to C and D,
    # 10: Common to A, B, and C, 11: Common to A, B, and D,
    # 12: Common to A, C, and D, 13: Common to B, C, and D,
    # 14: Common to A, B, C, and D

    # We need to map these to the order expected by our Venn4 drawing function
    return [
        var_part.iloc[0]["Fractions"],  # A only
        var_part.iloc[1]["Fractions"],  # B only
        var_part.iloc[2]["Fractions"],  # C only
        var_part.iloc[3]["Fractions"],  # D only
        var_part.iloc[4]["Fractions"],  # AB only
        var_part.iloc[5]["Fractions"],  # AC only
        var_part.iloc[6]["Fractions"],  # AD only
        var_part.iloc[7]["Fractions"],  # BC only
        var_part.iloc[8]["Fractions"],  # BD only
        var_part.iloc[9]["Fractions"],  # CD only
        var_part.iloc[10]["Fractions"],  # ABC only
        var_part.iloc[11]["Fractions"],  # ABD only
        var_part.iloc[12]["Fractions"],  # ACD only
        var_part.iloc[13]["Fractions"],  # BCD only
        var_part.iloc[14]["Fractions"]  # ABCD
    ]


def _draw_venn2(ax, values, labels, colors, fontsize):
    """Draw 2-set Venn diagram"""
    # Values: [A, B, AB]
    A, B, AB = values

    # Calculate circle positions and sizes
    r = 0.5
    center1 = (-0.2, 0)
    center2 = (0.2, 0)

    # Draw circles with transparency - 修复：使用facecolor而不是color
    circle1 = Circle(center1, r, fill=True, alpha=0.5, facecolor=colors[0],
                     edgecolor='black', linewidth=1)
    circle2 = Circle(center2, r, fill=True, alpha=0.5, facecolor=colors[1],
                     edgecolor='black', linewidth=1)

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Add labels
    ax.text(center1[0] - 0.3, center1[1], f"{labels[0]}\n{A:.3f}",
            ha='center', va='center', fontsize=fontsize - 1, fontweight='bold')
    ax.text(center2[0] + 0.3, center2[1], f"{labels[1]}\n{B:.3f}",
            ha='center', va='center', fontsize=fontsize - 1, fontweight='bold')
    ax.text(0, 0, f"{AB:.3f}",
            ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Set limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.7, 0.7)


def _draw_venn3(ax, values, labels, colors, fontsize):
    """Draw 3-set Venn diagram"""
    # Values: [A, B, C, AB, AC, BC, ABC]
    A, B, C, AB, AC, BC, ABC = values

    # Calculate circle positions (equilateral triangle)
    r = 0.4
    center1 = (0, 0.3)
    center2 = (-0.3, -0.2)
    center3 = (0.3, -0.2)

    # Draw circles - 修复：使用facecolor而不是color
    circle1 = Circle(center1, r, fill=True, alpha=0.5, facecolor=colors[0],
                     edgecolor='black', linewidth=1)
    circle2 = Circle(center2, r, fill=True, alpha=0.5, facecolor=colors[1],
                     edgecolor='black', linewidth=1)
    circle3 = Circle(center3, r, fill=True, alpha=0.5, facecolor=colors[2],
                     edgecolor='black', linewidth=1)

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Add labels
    ax.text(center1[0], center1[1] + 0.5, f"{labels[0]}\n{A:.3f}",
            ha='center', va='center', fontsize=fontsize - 1, fontweight='bold')
    ax.text(center2[0] - 0.5, center2[1] - 0.3, f"{labels[1]}\n{B:.3f}",
            ha='center', va='center', fontsize=fontsize - 1, fontweight='bold')
    ax.text(center3[0] + 0.5, center3[1] - 0.3, f"{labels[2]}\n{C:.3f}",
            ha='center', va='center', fontsize=fontsize - 1, fontweight='bold')

    # Add intersection values
    ax.text(-0.2, 0.1, f"{AB:.3f}", ha='center', va='center', fontsize=fontsize - 1)
    ax.text(0.2, 0.1, f"{AC:.3f}", ha='center', va='center', fontsize=fontsize - 1)
    ax.text(0, -0.1, f"{BC:.3f}", ha='center', va='center', fontsize=fontsize - 1)
    ax.text(0, 0, f"{ABC:.3f}", ha='center', va='center', fontsize=fontsize)

    # Set limits
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.7, 0.8)


def _draw_venn4(ax, values, labels, colors, fontsize):
    """Draw 4-set Venn diagram (simplified version)"""
    # Values: [A, B, C, D, AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD]
    A, B, C, D, AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD = values

    # Use ellipses for 4-set Venn (simplified)
    r = 0.35

    # Positions for 4 ellipses in a diamond pattern
    centers = [
        (0, 0.3),  # A - top
        (-0.3, 0),  # B - left
        (0.3, 0),  # C - right
        (0, -0.3)  # D - bottom
    ]

    # Draw ellipses - 修复：使用facecolor而不是color
    for i, center in enumerate(centers):
        ellipse = Ellipse(center, r, r, fill=True, alpha=0.4,
                          facecolor=colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(ellipse)

    # Add variable labels
    ax.text(centers[0][0], centers[0][1] + 0.5, f"{labels[0]}\n{A:.3f}",
            ha='center', va='center', fontsize=fontsize - 2, fontweight='bold')
    ax.text(centers[1][0] - 0.5, centers[1][1], f"{labels[1]}\n{B:.3f}",
            ha='center', va='center', fontsize=fontsize - 2, fontweight='bold')
    ax.text(centers[2][0] + 0.5, centers[2][1], f"{labels[2]}\n{C:.3f}",
            ha='center', va='center', fontsize=fontsize - 2, fontweight='bold')
    ax.text(centers[3][0], centers[3][1] - 0.5, f"{labels[3]}\n{D:.3f}",
            ha='center', va='center', fontsize=fontsize - 2, fontweight='bold')

    # Add some key intersection values (simplified for clarity)
    ax.text(-0.15, 0.15, f"{AB:.3f}", ha='center', va='center', fontsize=fontsize - 3)
    ax.text(0.15, 0.15, f"{AC:.3f}", ha='center', va='center', fontsize=fontsize - 3)
    ax.text(0, 0, f"{ABCD:.3f}", ha='center', va='center', fontsize=fontsize - 2, fontweight='bold')

    # Set limits
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)


def plot_comparison(results: List[RdaccaHpResult],
                    plot_perc: bool = True,
                    colors: Optional[List[str]] = None,
                    figsize: tuple = (12, 8),
                    title: Optional[str] = None,
                    **kwargs) -> plt.Figure:
    """
    Compare multiple rdacca_hp results in a single plot

    Parameters
    ----------
    results : list of RdaccaHpResult
        Multiple result objects to compare
    plot_perc : bool
        Whether to show percentages
    colors : list, optional
        Colors for different results
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not results:
        raise ValueError("At least one result is required")

    n_results = len(results)

    # Set up colors
    if colors is None:
        if _has_seaborn:
            colors = sns.color_palette("Set2", n_results)
        else:
            colors = [f'C{i}' for i in range(n_results)]

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_results, figsize=figsize, squeeze=False)
    axes = axes[0]  # Flatten

    for i, (result, ax, color) in enumerate(zip(results, axes, colors)):
        # Use the individual bar plot function for each result
        _plot_single_comparison(ax, result, plot_perc, color, i, **kwargs)

    # Set overall title
    if title is None:
        title = "Comparison of Hierarchical Partitioning Results"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig


def _plot_single_comparison(ax, result, plot_perc, color, idx, **kwargs):
    """Helper function to plot single result in comparison plot"""
    hier_part = result.hier_part

    if plot_perc:
        values = hier_part["I.perc(%)"].values
        ylabel = "% Individual effect"
    else:
        values = hier_part["Individual"].values
        ylabel = "Individual effect"

    variable_names = hier_part.index.tolist()

    # Sort by values
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = values[sorted_indices]
    sorted_names = [variable_names[i] for i in sorted_indices]

    # Create bar plot
    bars = ax.bar(range(len(values)), sorted_values, color=color, alpha=0.7,
                  edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, value in zip(bars, sorted_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(sorted_values),
                f'{value:.1f}' + ('%' if plot_perc else ''),
                ha='center', va='bottom', fontsize=10)

    # Customize subplot
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)

    # Set subplot title
    method, rtype = result.method_type
    ax.set_title(f"{method} ({rtype})", fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


# Add the plotting method to RdaccaHpResult class for convenience
def _rdaccahp_plot(self, plot_type="bar", **kwargs):
    """Convenience method to plot directly from result object"""
    return plot_rdaccahp(self, plot_type=plot_type, **kwargs)


# Attach the method to the class
RdaccaHpResult.plot = _rdaccahp_plot


# 添加requirements.txt建议
def check_dependencies():
    """Check if recommended dependencies are available"""
    missing = []

    if not _has_seaborn:
        missing.append("seaborn (optional, for better color palettes)")

    if missing:
        print("Recommended dependencies not found:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nYou can install them with: pip install seaborn")

    return len(missing) == 0


# Example usage and test function
def test_plotting():
    """Test plotting functions with example data"""
    try:
        # Create some example data for testing
        from .core import RdaccaHpResult

        # Mock hierarchical partitioning results
        hier_data = {
            'Unique': [0.2, 0.3, 0.1],
            'Average.share': [0.05, 0.08, 0.02],
            'Individual': [0.25, 0.38, 0.12],
            'I.perc(%)': [33.33, 50.67, 16.00]
        }
        hier_df = pd.DataFrame(hier_data, index=['Var1', 'Var2', 'Var3'])

        # Mock variation partitioning results for 3 variables
        var_data = {
            'Fractions': [0.2, 0.3, 0.1, 0.05, 0.02, 0.08, 0.02],
            '% Total': [28.57, 42.86, 14.29, 7.14, 2.86, 11.43, 2.86]
        }
        var_index = [
            'Unique to Var1', 'Unique to Var2', 'Unique to Var3',
            'Common to Var1 and Var2', 'Common to Var1 and Var3',
            'Common to Var2 and Var3', 'Common to Var1, Var2, and Var3'
        ]
        var_df = pd.DataFrame(var_data, index=var_index)

        # Create result object
        result = RdaccaHpResult(
            method_type=['RDA', 'adjR2'],
            total_explained_variation=0.75,
            hier_part=hier_df,
            var_part=var_df
        )

        print("Testing plotting functions...")

        # Test bar plot
        fig1 = plot_rdaccahp(result, plot_type='bar', plot_perc=False)
        plt.savefig('test_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Bar plot created: test_bar_plot.png")

        # Test bar plot with percentages
        fig2 = plot_rdaccahp(result, plot_type='bar', plot_perc=True)
        plt.savefig('test_bar_plot_perc.png', dpi=300, bbox_inches='tight')
        print("✓ Bar plot with percentages created: test_bar_plot_perc.png")

        # Test Venn diagram
        fig3 = plot_rdaccahp(result, plot_type='venn')
        plt.savefig('test_venn_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Venn diagram created: test_venn_plot.png")

        # Test convenience method
        fig4 = result.plot(plot_type='bar', color=['red', 'blue', 'green'])
        plt.savefig('test_convenience_plot.png', dpi=300, bbox_inches='tight')
        print("✓ Convenience method plot created: test_convenience_plot.png")

        plt.close('all')
        print("All plotting tests completed! 🎉")

    except Exception as e:
        print(f"Plotting test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_plotting()