import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, t
import pandas as pd


# The assumption of **independence** is satisfied because the user IDs are.
# The assumption of **normality** is relaxed due to the large sample size, supported by the **Central Limit Theorem**.
# The assumption of **equal variances** is met, as Levene’s test results show no significant difference in variances between the two groups, if is - use Welch's t-test with param equal_var=False.
def two_sides_ttest_with_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = True,
    confidence: float = 0.95,
    round_to: int = 3,
) -> dict:
    """
    Perform an independent t-test between two groups and compute a confidence interval for the difference in means.

    Parameters:
        group1 (np.ndarray): Data for the first group.
        group2 (np.ndarray): Data for the second group.
        equal_var (bool): If True, assume equal variance (Student's t-test).
                          If False, use Welch's t-test. Defaults to True.
        confidence (float): Confidence level for the confidence interval. Defaults to 0.95.

    Returns:
        Dict[str, float | Tuple[float, float]]: A dictionary containing:
            - "t_stat": The calculated t-statistic.
            - "p_value": The p-value of the test.
            - "mean_diff": The difference in means between the two groups.
            - "confidence_interval": A tuple with the lower and upper bounds of the confidence interval.
    """
    t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    mean_diff = mean1 - mean2

    se_diff = np.sqrt(var1 / n1 + var2 / n2)

    df = ((var1 / n1 + var2 / n2) ** 2) / (
        ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    )

    critical_t = t.ppf((1 + confidence) / 2, df)

    ci_lower = mean_diff - critical_t * se_diff
    ci_upper = mean_diff + critical_t * se_diff

    return {
        "t_stat": float(round(t_stat, round_to)),
        "p_value": float(round(p_value, round_to * 2)),
        "mean_diff": float(round(mean_diff, round_to)),
        "confidence_interval": (
            float(round(ci_lower, round_to)),
            float(round(ci_upper, round_to)),
        ),
    }


def print_two_sides_ttest_with_ci(results: dict) -> None:
    """
    Print the results of a t-test in a formatted manner.

    Parameters:
        results (dict): A dictionary containing t-test results.
                        Expected keys: 't_stat', 'p_value', 'mean_diff', 'confidence_interval'.

    Returns:
        None
    """
    print(f"T-statistic: {results['t_stat']}")
    print(f"P-value: {results['p_value']}")
    print(f"Mean Difference: {results['mean_diff']}")
    print(f"95% Confidence Interval: {results['confidence_interval']}")


# Check Assumptions
# Expected Frequencies > 5
# Independence of Observations
# Sufficiently Large Sample Size
def chi_square_test(
    df: pd.DataFrame, feature1: str, target: str, round_to: int = 3
) -> dict:
    """
    Perform a Chi-Square test of independence between a categorical feature and the target variable.

    Parameters:
        df (pd.DataFrame): The dataframe containing the categorical variables.
        feature1 (str): The name of the categorical feature (independent variable).
        target (str): The name of the target variable (dependent variable).

    Returns:
        dict: A dictionary containing the Chi-Square statistic, p-value, degrees of freedom,
              and expected frequencies.
    """
    contingency_table = pd.crosstab(df[feature1], df[target])

    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        "chi2_stat": float(round(chi2_stat, round_to)),
        "p_value": float(round(p_value, round_to * 2)),
        "degrees_of_freedom": dof,
        "expected_frequencies": expected,
    }


def print_chi_square_test(results: dict, feature: str, target: str) -> None:
    """
    Print the results of a test in a formatted manner.

    Parameters:
        results (dict): A dictionary containing test results.
        feature1 (str): The name of the categorical feature (independent variable).
        target (str): The name of the target variable (dependent variable).

    Returns:
        None
    """
    print(f"Chi-Square Test between {feature} and {target}:")
    print(f"Expected Frequencies:\n{results['expected_frequencies']}")
    print(f"Chi2 Statistic: {results['chi2_stat']}")
    print(f"P-value: {results['p_value']}")
    print(f"Degrees of Freedom: {results['degrees_of_freedom']}")
