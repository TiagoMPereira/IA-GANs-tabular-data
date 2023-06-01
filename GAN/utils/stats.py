import pandas as pd
from scipy.stats import kstest

def kolmogorov_smirnov(original: pd.DataFrame, generated: pd.DataFrame):

    statistical_results = pd.DataFrame()
    cols = list(set(original.columns).union(set(generated.columns)))
    for col in cols:
        stats = kstest(original[col], generated[col]).statistic
        statistical_results[col] = [stats]

    mean = statistical_results.mean(axis=1).values
    median = statistical_results.median(axis=1).values
    std = statistical_results.std(axis=1).values
    min = statistical_results.min(axis=1).values
    max = statistical_results.max(axis=1).values

    statistical_results["mean"] = mean
    statistical_results["median"] = median
    statistical_results["std"] = std
    statistical_results["min"] = min
    statistical_results["max"] = max

    return statistical_results