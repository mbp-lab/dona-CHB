from pathlib import Path
from modules.utils import significance_asterisk
import os

data_reports_path=Path('../../reports/numbers_updated.dat') 

def calculate_wilcoxon(pre,post):
    """
    Performs the Wilcoxon signed-rank test on two paired samples and computes additional statistics.

    This function compares two related samples using a custom implementation of the Wilcoxon signed-rank test,
    then cross-validates the results against SciPy's built-in `wilcoxon` function. It also computes the 
    standardized test statistic (Z value) and the effect size.

    Parameters:
    ----------
    pre : array-like
        The pre-intervention or baseline sample values.
    post : array-like
        The post-intervention or follow-up sample values.

    Returns:
    -------
    w_python : float
        Test statistic calculated using SciPy's `wilcoxon` function.
    p_python : float
        Two-sided p-value from SciPy's `wilcoxon` function.
    Z : float
        Standardized test statistic based on normal approximation.
    effect_size : float
        Effect size calculated as |Z| divided by the square root of the number of observations.

    Notes:
    -----
    - Differences equal to zero are excluded from the rank calculation.
    - The function prints a warning if the custom and built-in results differ by more than 0.01.
    """
    
    from scipy.stats import wilcoxon,rankdata,norm
    import numpy as np
    diff = post-pre
    abs_diff = np.abs(diff)

    # Filter out zero differences
    non_zero_diff_indices = abs_diff != 0
    non_zero_diff = diff[non_zero_diff_indices]
    abs_non_zero_diff = abs_diff[non_zero_diff_indices]

    # Rank the absolute non-zero differences
    ranks = rankdata(abs_non_zero_diff)

    # Initialize rank array
    ranked_diffs = np.zeros_like(diff, dtype=float)
    ranked_diffs[non_zero_diff_indices] = ranks

    # Sum of positive and negative ranks
    positive_ranks = np.sum(ranked_diffs[diff > 0])
    negative_ranks = np.sum(ranked_diffs[diff < 0])
    W = min(positive_ranks, negative_ranks)

    # Number of pairs
    N = len(ranked_diffs[non_zero_diff_indices])

    # Calculate mean and standard deviation of W under null hypothesis
    mean_rank = N * (N + 1) / 4
    std_rank = np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    
    # Calculate Z value
    Z = (W - mean_rank) / std_rank

    # Calculte effect size
    effect_size = np.abs(Z/np.sqrt(len(diff)))

    # Calculate two-sided p-value using normal distribution
    p_value = 2 * (1 - norm.cdf(np.abs(Z)))

    w_python,p_python = wilcoxon(pre,post)
    if W != w_python or abs(p_value-p_python)>0.01:
        print('Custom and built-in functions for Wilcoxon test do not agree')
    return w_python, p_python, Z, effect_size


def summarize_score_comparison(analysis, question_code, data, W, p_value, Z, effect_size,save = True,display=True):
    """
    Summarizes and reports the results of a Wilcoxon test and descriptive statistics for pre/post scores.

    This function compiles test statistics and summary metrics for a given question from a paired test analysis.
    It optionally displays the results in tabular form and saves them to a CSV report file.

    Parameters:
    ----------
    analysis : str
        A label for the analysis (used as a prefix in output metric names).
    question_code : str
        The base identifier for the question, which is used to extract column names from `data`.
        Columns expected are: `{question_code}_pre`, `{question_code}_post`, and `{question_code}_diff`.
    data : pandas.DataFrame
        DataFrame containing the relevant pre, post and diff columns for the question.
    W : float
        Wilcoxon test statistic.
    p_value : float
        Two-sided p-value from the Wilcoxon test.
    Z : float
        Standardized test statistic (Z-score).
    effect_size : float
        Effect size computed from the Z-score.
    save : bool, optional (default=True)
        Whether to save the summary results to a CSV file.
    display : bool, optional (default=True)
        Whether to print the summary results to the console.

    Returns:
    -------
    None
        The function has no return value. It prints and/or saves the summary results.

    Notes:
    -----
    - Uses a helper function `significance_asterisk(p)` from the `utils` module to annotate p-values.
    - Assumes the existence of a global variable `data_reports_path` that defines the output CSV file path.
    - If `data_reports_path` exists, it will update the file with new values (dropping old duplicates).
    """
    
    from tabulate import tabulate
    from modules.utils import significance_asterisk
    import pandas as pd
    
    # Calculate the percentage of people who have changed and not changed their responses between pre and post assessments
    zero_proportion = round(len(data[f'{question_code}_diff'][data[f'{question_code}_diff'] == 0]) / len(data[f'{question_code}_diff']) * 100, 3)
    non_zero_proportion = round(100 - zero_proportion, 3)
    
    results = [
        [f"{analysis}-Wilcoxon_Stat", round(W, 3)],
        [f"{analysis}-Wilcoxon_p", significance_asterisk(p_value)],
        [f"{analysis}-Wilcoxon_Z", round(Z, 3)],
        [f"{analysis}-ES", round(effect_size, 3)],
        [f"{analysis}-Zero_Diff", f"{zero_proportion}"],
        [f"{analysis}-Non-Zero_Diff", f"{non_zero_proportion}"] ]
    
    desc_stats = []
    for stat in ["Mean", "Median", "Std", "Min", "Max"]:
        for test in ["pre", "post", "diff"]:
            value = getattr(data[f"{question_code}_{test.split('-')[0]}"], stat.lower())()
            desc_stats.append([f"{analysis}-{test}-{stat}", round(value, 3)])
    
    df_results = pd.DataFrame(results, columns=["Metric", "Value"])
    df_stats = pd.DataFrame(desc_stats, columns=["Metric", "Value"])
    df_combined = pd.concat([df_results, df_stats], ignore_index=True)
    if display:
         print(tabulate(df_combined.values.reshape(-1, 2), tablefmt='plain'))
    if save:
       if os.path.exists(data_reports_path):
          df_existing = pd.read_csv(data_reports_path, sep=',', engine="python", header=None, names=["Metric", "Value"], 
                    skipinitialspace=True)
          df_combined = pd.concat([df_existing, df_combined]).drop_duplicates(subset=["Metric"], keep='last')
       else: 
           pass
    df_combined.to_csv(data_reports_path, index=False, header=False, sep=',',  quoting=3)

def calculate_mcnemar(first, second):
    """
    Performs McNemar's test on paired binary outcomes and calculates the odds ratio.

    This function compares two related binary variables (e.g., pre/post answers )
    to test for significant changes in proportions using McNemar's test. It also
    computes the odds ratio of changes in one direction versus the other.

    Parameters:
    ----------
    first : list or array-like
        Binary outcomes (0 or 1) for the first condition (e.g., pre-intervention).
    second : list or array-like
        Binary outcomes (0 or 1) for the second condition (e.g., post-intervention).

    Returns:
    -------
    statistic : float
        The McNemar test statistic.
    pvalue : float
        The p-value associated with the test.
    odds_ratio : float
        The odds ratio of discordant pairs (0→1 vs. 1→0 transitions).

    Notes:
    -----
    - Uses the asymptotic version of McNemar's test (`exact=False`) with continuity correction.
    - The odds ratio is calculated as (# of 0→1 transitions) / (# of 1→0 transitions).
      - If one of these counts is 0, the odds ratio is set to `inf` or `0` accordingly to avoid division by zero.
    """
    
    from statsmodels.stats.contingency_tables import mcnemar

    # Validate inputs
    if len(first) != len(second):
        raise ValueError("Inputs must have the same length.")
    if not set(first).issubset({0, 1}) or not set(second).issubset({0, 1}):
        raise ValueError("Inputs must be binary (0 or 1 only).")
        
    # Create contingency table elements
    both_selected = sum(1 for a, b in zip(first, second) if a == 1 and b == 1)
    both_zero = sum(1 for a, b in zip(first, second) if a == 0 and b == 0)
    first_selected = sum(1 for a, b in zip(first, second) if a == 1 and b == 0)  # Changed from 1 to 0
    second_selected = sum(1 for a, b in zip(first, second) if a == 0 and b == 1)  # Changed from 0 to 1

    # Contingency table
    contingency_table = [[both_selected, first_selected],
                         [second_selected, both_zero]]

    # McNemar test
    test = mcnemar(contingency_table, exact=False, correction=True)

    # Calculate Odds Ratio 
    if first_selected == 0 or second_selected == 0:
        odds_ratio = float('inf') if second_selected > 0 else 0  # Handle edge cases
    else:
        odds_ratio = second_selected / first_selected

    return test.statistic, test.pvalue, odds_ratio


def compute_spearman_correlations(data, question_code, quantitative_default):
    """
    Computes Spearman rank-order correlations between question scores and a quantitative variable,
    along with confidence intervals for the difference between pre and post correlations.

    This function calculates Spearman correlations for:
    - pre vs. quantitative variable
    - post vs. quantitative variable
    - diff vs. quantitative variable
    - pre vs. post 
    
    It also compares the pre and post correlations using Zou's confidence interval method.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing the question-related columns (e.g., `{question_code}_pre`, `_post`, `_diff`)
        and the quantitative variable to correlate with.
    question_code : str
        The base name of the question column, used to generate full column names.
    quantitative_default : str
        The name of the quantitative variable column to correlate against (e.g., PDI, rGini index. etc).
    Returns:
    -------
    results : dict
        Dictionary containing Spearman correlation results (correlation coefficient and p-value)
        for pre, post, diff vs. the quantitative variable, and pre vs. post.
        Keys: `"pre_spearman"`, `"post_spearman"`, `"diff_spearman"`, `"pre_post_r"`.
    lower_zou : float
        Lower bound of the confidence interval for the difference between pre and post correlations (Zou's method).
    upper_zou : float
        Upper bound of the confidence interval for the difference between pre and post correlations (Zou's method).

    Notes:
    -----
    - Assumes the existence of a helper function `compare_correlations_zou(r1, r2, r12, n)` that computes 
      the confidence interval for the difference between two dependent correlations using Zou's method.
    - The `pre_post_r` value (correlation between pre and post scores) is required for the dependency correction.
    """
    from scipy.stats import spearmanr
    results = {
        "pre_spearman": spearmanr(data[f'{question_code}_pre'], data[quantitative_default]),
        "post_spearman": spearmanr(data[f'{question_code}_post'], data[quantitative_default]),
        "diff_spearman": spearmanr(data[f'{question_code}_diff'], data[quantitative_default]),
        "pre_post_r": spearmanr(data[f'{question_code}_pre'].tolist(), data[f'{question_code}_post'].tolist())
    }
    
    
    return results

def summarize_spearman_results(analysis, quantitative_default, results, zou_lower, zou_upper,t_steiger_2,p_steiger_2,t_steiger_1,p_steiger_1):
    import pandas as pd
    from tabulate import tabulate
    import numpy as np
    from modules.utils import significance_asterisk
    variables = {
        f"{analysis}-pre-{quantitative_default}-spearman-r": np.round(results["pre_spearman"][0],2),
        f"{analysis}-pre-{quantitative_default}-spearman-p": significance_asterisk(results["pre_spearman"][1]),
        f"{analysis}-post-{quantitative_default}-spearman-r": np.round(results["post_spearman"][0],2),
        f"{analysis}-post-{quantitative_default}-spearman-p": significance_asterisk(results["post_spearman"][1]),
        f"{analysis}-diff-{quantitative_default}-spearman-r": np.round(results["diff_spearman"][0],2),
        f"{analysis}-diff-{quantitative_default}-spearman-p": significance_asterisk(results["diff_spearman"][1]),
        f"{analysis}-{quantitative_default}-zou-upper": np.round(zou_upper,2),
        f"{analysis}-{quantitative_default}-zou-lower": np.round(zou_lower,2),
        f"{analysis}-{quantitative_default}-steiger-t2": np.round(t_steiger_2,2),
        f"{analysis}-{quantitative_default}-steiger-p2": significance_asterisk(p_steiger_2),
        f"{analysis}-{quantitative_default}-steiger-t1": np.round(t_steiger_1,2),
        f"{analysis}-{quantitative_default}-steiger-p1": significance_asterisk(p_steiger_1),

    }
    
    df = pd.DataFrame(variables.items(), columns=["Metric", "Value"])
    print(tabulate(df, headers='keys', tablefmt='grid'))
    if os.path.exists(data_reports_path):
        df_existing = pd.read_csv(data_reports_path, sep=',', engine="python", header=None, names=["Metric", "Value"], quotechar='"', skipinitialspace=True)
        df_combined = pd.concat([df_existing, df]).drop_duplicates(subset=["Metric"], keep='last')
    else:
        df_combined = df
    df_combined.to_csv(data_reports_path, index=False, header=False, sep=',',  quoting=3,escapechar='\\')

def compare_correlations_zou(xy, xz, yz, n, conf_level=0.95):
    """
    Function for calculating the statistical significant differences between two dependent correlation
    coefficients.
    The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
    Credit goes to the authors of above mentioned packages!
    Author: Philipp Singer (www.philippsinger.info)
    
    ------------------------------------------------------------
    README.md:
    CorrelationStats
    This Python script enables you to compute statistical significance tests on both dependent and independent correlation coefficients. For each 
    case two methods to choose from are available.
    
    For details, please refer to: http://www.philippsinger.info/?p=347
    #copied from on 28/02/2025 from https://github.com/MicrosoftResearch/Azimuth/blob/master/azimuth/corrstats.py
    """    
    import numpy as np
    from scipy.stats import t, norm
    from math import atanh, pow
    from numpy import tanh
    
    def rz_ci(r, n, conf_level = 0.95):
        zr_se = pow(1/(n - 3), .5)
        moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
        zu = atanh(r) + moe
        zl = atanh(r) - moe
        return tanh((zl, zu))
    
    def rho_rxy_rxz(rxy, rxz, ryz):
        num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
        den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
        return num/float(den)
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param conf_level: confidence level, only works for 'zou' method
    @return: lower and upper values of the confidence interval
    """
    L1 = rz_ci(xy, n, conf_level=conf_level)[0]
    U1 = rz_ci(xy, n, conf_level=conf_level)[1]
    L2 = rz_ci(xz, n, conf_level=conf_level)[0]
    U2 = rz_ci(xz, n, conf_level=conf_level)[1]
    rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
    lower = xy - xz - pow((pow((xy - L1), 2) + pow((U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
    upper = xy - xz + pow((pow((U1 - xy), 2) + pow((xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
    return lower, upper

def compare_correlations_steiger(xy,xz, yz,n,twotailed=True,conf_level=0.95):
    """
    Function for calculating the statistical significant differences between two dependent correlation
    coefficients.
    The Steiger method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
    Credit goes to the authors of above mentioned packages!
    Author: Philipp Singer (www.philippsinger.info)
    
    ------------------------------------------------------------
    README.md:
    CorrelationStats
    This Python script enables you to compute statistical significance tests on both dependent and independent correlation coefficients. For each 
    case two methods to choose from are available.
    
    For details, please refer to: http://www.philippsinger.info/?p=347
    #copied from on 28/02/2025 from https://github.com/MicrosoftResearch/Azimuth/blob/master/azimuth/corrstats.py
    """    

    import numpy as np
    from scipy.stats import t, norm
    from math import atanh, pow
    from numpy import tanh

    def rz_ci(r, n, conf_level = 0.95):
        zr_se = pow(1/(n - 3), .5)
        moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
        zu = atanh(r) + moe
        zl = atanh(r) - moe
        return tanh((zl, zu))

    def rho_rxy_rxz(rxy, rxz, ryz):
        num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
        den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
        return num/float(den)
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    d = xy - xz
    determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
    p = 1 - t.cdf(abs(t2), n - 2)

    if twotailed:
        p *= 2

    return t2, p

def spearman_ci(x, y, n_bootstrap=10000, ci=95, random_state=None):
    
    """
    Compute Spearman's correlation and its bootstrap confidence interval.

    Parameters:
    x, y : array-like (Pandas Series or NumPy arrays)
        Input data vectors.
    n_bootstrap : int, optional
        Number of bootstrap samples (default is 10,000).
    ci : float, optional
        Confidence level (default is 95% for a 95% CI).
    random_state : int, optional
        Seed for reproducibility.

    Returns:
    spearman_rho : float
        Spearman's correlation coefficient.
    ci_lower, ci_upper : float
        Lower and upper bounds of the confidence interval.
    """
    import numpy as np
    import scipy.stats as stats
    import pandas as pd

    np.random.seed(random_state)
    n = len(x)

    x = np.array(x)
    y = np.array(y)

    bootstrapped_rhos = []

    # Compute the original Spearman correlation
    spearman_rho, _ = stats.spearmanr(x, y)

    # Bootstrapping
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_rho, _ = stats.spearmanr(x[indices], y[indices])
        bootstrapped_rhos.append(boot_rho)

    # Compute the confidence interval
    lower_bound = np.percentile(bootstrapped_rhos, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_rhos, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound

def lowess_smooth(x, y, frac=1):
    """
    Applies LOWESS (Locally Weighted Scatterplot Smoothing) to smooth a set of data points for trend visualization.

    Parameters:
    ----------
    x : array-like
        The independent variable.
    y : array-like
        The dependent variable.
    frac : float, optional (default=1)
        The fraction of the data used when estimating each y-value. 
        Smaller values result in less smoothing (more local detail), 
        while larger values result in more smoothing (broader trends).

    Returns:
    -------
    x_smoothed : ndarray
        The sorted x-values used in the smoothed output.
    y_smoothed : ndarray
        The smoothed y-values corresponding to `x_smoothed`.

    Notes:
    -----
    - `frac=1` uses the full dataset for each local regression, which may oversmooth the data.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(y, x, frac=frac)
    return smoothed[:, 0], smoothed[:, 1]

