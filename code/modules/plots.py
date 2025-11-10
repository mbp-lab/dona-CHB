import matplotlib.pyplot as plt
from pathlib import Path

# Where to save figures from the analysis
data_reports_path=Path('../../reports/numbers_updated.dat') 
figpath = Path('../../reports')
pre_color = 'gray'
post_color = 'slateblue'
default_color = 'blue'

def set_aspect(ax):
    """
    Sets the aspect ratio of the axes to match the data scale.
    
    Parameters:
    ax : matplotlib.axes.Axes
        The axis to adjust.
    """

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))


def plot_score_diff_hist(data,question_code,analysis,display=True):
    """
    Plots histograms of pre, post as wel as of the difference scores for a given question.

    Parameters:
    data : DataFrame
        Dataset containing pre, post, and diff columns.
    question_code : str
        Prefix for column names (e.g., 'Q1').
    analysis : str
        Label used in saved filenames.
    display : bool
        If False, closes the plot after saving.
    """
    fig, axes = plt.subplots(1,2,figsize=(6,6))
    ax = axes[0]
    ax1 = axes[1]
    
    ax.hist(data[f'{question_code}_pre'],color=pre_color,edgecolor='k',alpha=.7,label='Pre')
    ax.hist(data[f'{question_code}_post'],color=post_color,edgecolor='k',alpha=.7,label='Post')

    ax.legend(prop={'size':14},loc='best')
    ax.set_xlabel('Agreement Score',fontsize=20)
    ax.set_ylabel('Participant Count',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(1,7)
    set_aspect(ax)

    bins = range(min(data[f'{question_code}_diff']), max(data[f'{question_code}_diff']) + 2)
    ax1.hist(data[f'{question_code}_diff'],bins=bins, align='left',edgecolor='k')
    ax1.set_xlim(-7,7)
    ax1.set_ylim(0,40)
    
    ax1.set_xlabel('Post-Pre Score',fontsize=20)
    ax1.set_ylabel('Participant count',fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    set_aspect(ax1)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-hist.png'),bbox_inches='tight',dpi=500)
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-hist.svg'),bbox_inches='tight',dpi=500)
    if not display:
        plt.close(fig)


def plot_heatmap(data, question_code,quantitative_columns, analysis):
    """
    Plots heatmaps of median quantitative values by pre/post score pairs .

    Parameters:
    data : DataFrame
        Data with pre, post, and quantitative columns.
    question_code : str
        Prefix for subjective score columns.
    analysis : str
        Analysis type ('dominance' or 'equality') for colormap and value range.
    quantitative_columns : list of str
        Objective metric  columns to visualize with median values in heatmaps.
    """
    import math
    import numpy as np
    fig, axes = plt.subplots(1, len(quantitative_columns), figsize=(10, 8))
    for ind, var in enumerate(quantitative_columns):
        ax = axes[ind]
        
        # Prepare data
        heatmap_data = data.groupby([f'{question_code}_post', f'{question_code}_pre'])[var].median().unstack()
        counts = data.groupby([f'{question_code}_post', f'{question_code}_pre']).size().unstack()
        
        # Plot heatmap
        if analysis == 'dominance':
            im = ax.imshow(heatmap_data, cmap='bwr', aspect='auto', vmin=0.1, vmax=0.9)
        elif analysis == 'equality':
            im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        title = " ".join(var.split()[:-1])
        ax.set_title(title, fontsize=16)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
        ax.invert_yaxis()
        
        # Add counts as text labels
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                count = counts.iloc[i, j]
                if not math.isnan(count):
                    ax.text(j, i, f'N={int(count)}', ha='center', va='center', color='black',fontsize=12)
        
        # Add grid-like patches
        for i in range(7):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', lw=2))
        
        # Set axes properties
        ticks = np.linspace(-0.5, 6.5, 7)
        ax.set_xlim(1, 6.5)
        ax.set_ylim(1, 6.5)
        ax.set_xlabel('Pre Score', fontsize=16)
        ax.set_ylabel('Post Score', fontsize=16)
        ax.set_xticks(ticks, np.linspace(1, 7, 7, dtype=int))
        ax.set_yticks(ticks, np.linspace(1, 7, 7, dtype=int))
        ax.tick_params(axis='both', which='major', labelsize=14)
        set_aspect(ax)
    
    # Final figure adjustments
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    
    # Save figure
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-obj-heatmap.png'), bbox_inches='tight', dpi=500)
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-obj-heatmap.svg'), bbox_inches='tight', dpi=500)


def plot_corr_differences(data,question_code,variables,analysis):
    """
    Plots Spearman correlations between question responses and target variables
    across multiple subgroups, highlighting significant changes from pre to post.

    Parameters:
    ----------
    data : DataFrame
        The full dataset containing pre, post and diff scores along with target variables.
    question_code : str
        Base column name for question scores, assuming columns like 'question_pre', 'question_post', etc.
    variables : list of str
        List of variable names to correlate with pre/post/diff scores.
    analysis : str
        Identifier used in figure titles and output file names.

    Notes:
    -----
    - Uses subgroups based on the magnitude of change: 'all', 'any', 'one', and 'large'.
    - Computes confidence intervals and tests for significant changes using Zou's method.
    - Saves results to disk using `save_variables()` and highlights significant differences with asterisks.
    """
    from scipy.stats import spearmanr
    from modules.tests import compare_correlations_zou, spearman_ci
    from modules.utils import get_outlier_bounds
    
    import numpy as np
    from modules.utils import save_variables, significance_asterisk
    
    # Variables and Colors
    colors = {'pre': pre_color, 'post': post_color, 'diff': default_color}
    
    # X-axis spacing
    x_spacing = 0.8
    x_positions = np.linspace(0, x_spacing * (len(variables) - 1), len(variables))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    x_ticks, x_labels = [], []
    
    # Loop through variables first
    for var_idx, var in enumerate(variables):
        x_center = x_positions[var_idx]
        ax.text(x_center, 1.08, var, fontsize=13, fontweight="bold", ha="center", transform=ax.get_xaxis_transform())
        correlation_data = data.copy()
        lower_bound, upper_bound = get_outlier_bounds(correlation_data[var])
        correlation_data[f'{var}-outlier'] = ((correlation_data[var] < lower_bound) | (correlation_data[var] > upper_bound)).astype(int)

        # Define subsets based on response differences

        datasets = {
        "all": correlation_data,
        "any": correlation_data[correlation_data[f'{question_code}_diff'].abs() != 0],
        "large": correlation_data[correlation_data[f'{question_code}_diff'].abs() > 1]}
        dataset_positions = np.linspace(-0.15, 0.15, len(datasets))
        
        for dataset_idx, (dataset_label, dat) in enumerate(datasets.items()):
            plot_data = dat[dat[f'{var}-outlier'] == 0]
            save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-N', len(plot_data))
            x_base = x_center + dataset_positions[dataset_idx]
            spearman_values = {}
            pre_post_sub_r, _ = spearmanr(plot_data[f'{question_code}_pre'], plot_data[f'{question_code}_post'])
            corr = []
    
            for prefix in ['pre', 'post', 'diff']:
                r, p = spearmanr(plot_data[f'{question_code}_{prefix}'], plot_data[var])
                spearman_lower, spearman_upper = spearman_ci(plot_data[f'{question_code}_{prefix}'], plot_data[var])
    
                corr.append(r)
                if prefix != 'diff':
                    marker = 'o' if p < 0.05 else '.'
                    ax.scatter(x_base, r, color=colors[prefix], marker=marker, s=110, linewidth=1.2)
                spearman_values[prefix] = r
    
                # Save results
                save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-{prefix}-spearman-lower', np.round(spearman_lower, 3))
                save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-{prefix}-spearman-upper', np.round(spearman_upper, 3))
                save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-{prefix}-spearman-r', np.round(r, 3))
                save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-{prefix}-spearman-p', significance_asterisk(p),rounding=False)
    
            lower, upper = compare_correlations_zou(corr[1], corr[0], pre_post_sub_r, len(dat))
            save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-zou-lower', lower)
            save_variables(data_reports_path, f'{analysis}-{dataset_label}-{var}-zou-upper', upper)
    
            if not (lower <= 0 <= upper):
                ax.text(x_base, max(spearman_values.values()) + 0.1, '*', fontsize=14, fontweight="bold", ha="center", color='black')
    
            x_ticks.append(x_base)
            x_labels.append(f'{dataset_label}, N={len(plot_data)}')
            ax.plot([x_base, x_base], [spearman_values['pre'], spearman_values['post']], color='gray', linestyle='-', linewidth=1.5)
    
    # Formatting the plot
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=14, rotation=45, ha="right")
    ax.set_ylabel("Spearman Correlation", fontsize=15, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.linspace(-1, 1, 5))
    fig.tight_layout(pad=1.0)
    
    # Save and show
    plt.show()
    fig.savefig(Path(f'{figpath}/{analysis}-corrs.svg'), bbox_inches='tight', dpi=500)


def plot_corr_raw(correlation_data,outliers,question_code,stage,analysis,quantitative,color,label='',ylim=None):
    """
    Plots raw scatter data and a LOWESS-smoothed trend between a question score and a quantitative variable.
    
    Parameters:
    -----------
    sample_data : DataFrame
        The dataset containing the score and quantitative columns.
    question_code : str
        Prefix for subjective score column names, used to construct column names like 'question_pre'.
    stage : str
        One of 'pre', 'post', or 'diff' — determines which score column to plot.
    analysis : str
        Identifier for the analysis, used in the saved file name.
    quantitative : str
        The name of the objective metric to correlate with the score.
    color : str
        Color for the scatterplot points.
    label : str, optional
        Additional label to include in the saved filename (default: '').
    ylim : tuple or None, optional
        Y-axis limits. If None, limits are inferred. If provided, may trigger horizontal reference lines.
    
    Returns:
    --------
    None
        The function generates and saves the plot as an SVG file.
    """
    from modules.tests import lowess_smooth
    from scipy.stats import spearmanr
    import numpy as np
    fig, ax = plt.subplots(figsize=(4,4))  # Wider for better readability
    ax.plot(correlation_data[f'{question_code}_{stage}'], correlation_data[quantitative], '.', color=color, markersize=14,alpha=0.6)
    ax.plot(outliers[f'{question_code}_{stage}'], outliers[quantitative], '.', color=color, marker='x',markersize=6,alpha=0.6)
    x_smooth, y_smooth = lowess_smooth(correlation_data[f'{question_code}_{stage}'], correlation_data[quantitative])
    ax.plot(x_smooth, y_smooth, '-', color='red',linewidth=3)
    
    # Spearman correlation
    spearman_corr_all, spearman_p_all = spearmanr(correlation_data[f'{question_code}_{stage}'], correlation_data[quantitative])

    # Axis labels and limits with increased font sizes
    ax.set_ylabel(quantitative,fontsize=26)

    if stage == 'diff':
        ax.set_xlim(-7,7) 
        ax.set_xticks(np.linspace(-6,6,5))
        ax.set_xlabel('Post-Pre Score', fontsize=26)
    else:
        ax.set_xlim(0.9,7.1)
        # Set x-ticks explicitly to avoid rounding issues
        ax.set_xticks(np.arange(1, 8, 1))
        ax.set_xlabel('Agreement Score', fontsize=26)
    ax.set_title(f'N={len(correlation_data)}',fontsize=22)
    try:
        if ylim[0] < 0.5 < ylim[1]:
            ax.set_ylim(0.0,1)
            # Set exactly three y-ticks
            ax.set_yticks([0.0, 0.5, 1])    
            ax.plot(ax.get_xlim(),[0.5,0.5],'k--')
            ax.set_ylim(ylim)
        elif  ylim[0] < 0 < ylim[1]:
            ax.plot(ax.get_xlim(),[0.0,0.0],'k--')
            ax.set_ylim(ylim)
    except:
        pass
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)

    # Adjust layout and save the figure
    fig.tight_layout()
    fig.savefig(Path(f'{figpath}/{analysis}-{label}-corr-{stage}.svg'), bbox_inches='tight', dpi=500)

def plot_bin_selection(data,bin_columns,analysis,name=''):
    import numpy as np
    from modules.tests import calculate_mcnemar
    from modules.utils import save_descriptive_stats,save_variables, significance_asterisk
    bin_labels  = [col.replace('speed-', '') for col in bin_columns]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 1.4
    x_pos = np.linspace(1, len(bin_columns) * width * 3.5, len(bin_columns))

    # Plotting and statistical testing for each bin
    for ind, col in enumerate(bin_columns): 
        # Get pre/post data for this bin
        pre_data = data[f'{col}_pre']
        post_data = data[f'{col}_post']
    
        # Plot bars
        pre_bar = ax.bar(x_pos[ind], pre_data.sum(), width, color=pre_color, edgecolor='k')
        post_bar = ax.bar(x_pos[ind] + width, post_data.sum(), width, color=post_color, edgecolor='k')
    
        # McNemar test and Bonferroni correction
        stat, p_value, odds_ratio = calculate_mcnemar(pre_data, post_data)
        p_corrected = p_value * len(bin_columns)
    
        # Save statistics
        save_variables(data_reports_path, f'{analysis}-{col}-{name}-mcnemar-odds-ratio', odds_ratio)
        save_variables(data_reports_path, f'{analysis}-{col}-{name}-mcnemar-stat', stat)
        save_variables(data_reports_path, f'{analysis}-{col}-{name}-mcnemar-p', significance_asterisk(p_value), rounding=False)
        save_variables(data_reports_path, f'{analysis}-{col}-{name}-mcnemar-p-corrected', significance_asterisk(p_corrected), rounding=False)
    
        # Annotate significant differences
        if p_corrected < 0.05: 
            ax.plot([x_pos[ind] - width / 2, x_pos[ind] + width / 2], [37, 37], color='black')
            ax.text(x_pos[ind], 38, f'p={p_corrected:.3f}', ha='center', va='bottom', color='black', fontsize=12)
    
            # Add significance stars
            if p_corrected < 0.001:
                ax.text(x_pos[ind], 40, '***', ha='center', va='bottom', color='red', fontsize=14)
            elif p_corrected < 0.01:
                ax.text(x_pos[ind], 40, '**', ha='center', va='bottom', color='red', fontsize=14)
            else:
                ax.text(x_pos[ind], 40, '*', ha='center', va='bottom', color='red', fontsize=14)
    
    # Final plot formatting
    ax.set_ylabel('Participant count', fontsize=16)
    ax.set_xlabel('Bin', fontsize=16)
    ax.set_xticks(x_pos + width * 0.5)
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(0, 45)
    ax.set_xlim(x_pos[0] - 2 * width, x_pos[-1] + 3 * width)
    ax.tick_params(axis='both', which='major', labelsize=16, rotation=15)
    ax.legend([pre_bar, post_bar], ['Pre', 'Post'], loc='best', prop={'size': 14})
    
    # Finalize and save
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-bars-{name}.png'))
    fig.savefig(Path(f'{figpath}/{analysis}-pre-post-bars-{name}.svg'))
    plt.show()

