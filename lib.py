import numpy as np
from copy import copy
from IPython.display import display, Markdown
import warnings
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list


###
# Setup
###
def custom_formatwarning(msg, category, filename, lineno, line=None):
    """
    This suppresses the directory when the warning is called
    """
    return f'{category.__name__}: {msg}\n'


warnings.formatwarning = custom_formatwarning

####################################################################################

# Constants
NUM_DAYS = 252

# Constants
MOM_SPEEDS = {
    2: (1, 3),
    3: (2, 6),
    4: (4, 12),
    5: (8, 24),
    6: (16, 48),
    7: (32, 96),
    8: (64, 192),
    9: (128, 384)
}


####################################################################################

# Helper functions

def display_markdown(text):
    display(Markdown(text))


def cluster_matrix(matrix):
    """
    Cluster correlation matrix. ChatGPT made this
    """
    # Step 2: Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - np.abs(matrix)

    # Step 3: Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    # Step 4: Get the order of rows and columns according to the hierarchical clustering result
    sorted_indices = leaves_list(linkage_matrix)

    # Step 5: Reorder the correlation matrix
    sorted_corr_matrix = matrix.iloc[sorted_indices, sorted_indices]
    return sorted_corr_matrix


# Accounting functions
def sharpe(returns):
    return np.sqrt(NUM_DAYS) * returns.mean() / returns.std()


def appraisal(target_sharpe, benchmark_sharpe, correlation):
    return (target_sharpe - correlation * benchmark_sharpe) / np.sqrt(1 - correlation ** 2)


def market_holding_period(signal):
    """
    Computes the average amount of days we are long or short
    """
    # Transform signal into long (+1) or short (-1)
    signal = copy(signal.values)
    long_short = list(2 * (signal > 0.0).astype(float) - 1)

    # Count the number of consecutive days we are long or short
    counts = []
    while long_short:
        sign = long_short[-1]
        count = 0
        while long_short and (long_short[-1] == sign):
            count += 1
            long_short.pop()
        counts.append(count)
    return np.mean(counts)


def holding_period(signals):
    """
    The holding period is the average across all markets
    """
    market_holding_periods = []
    for market in signals.columns:
        signal = signals[market].dropna()
        market_holding_periods.append(market_holding_period(signal))
    return round(np.mean(market_holding_periods), 2)


####################################################################################

# Momentum

# Helper
def halflife(n):
    if n == 1:
        return 0.000001
    else:
        return np.log(0.5) / np.log(1 - (1 / n))


# Volatility
def daily_volatility(returns, lookback=50, min_periods=25):
    """
    min periods is in units of observations e.g. weeks or days
    """
    if min_periods is None:
        min_periods = lookback
    vol = returns.ewm(halflife=halflife(lookback), min_periods=min_periods).std()
    return vol


class Momentum:
    def __init__(self, fast, slow, return_vol_lookback=50, xover_vol_lookback=250, return_vol_min_periods=None,
                 xover_min_periods=None):
        if return_vol_min_periods is None:
            return_vol_min_periods = return_vol_lookback // 2
        if xover_min_periods is None:
            xover_min_periods = xover_vol_lookback // 2

        self.fast = fast
        self.slow = slow
        self.return_vol_lookback = return_vol_lookback
        self.return_vol_min_periods = return_vol_min_periods

        self.xover_vol_lookback = xover_vol_lookback
        self.xover_min_periods = xover_min_periods

    def __call__(self, price):
        returns = price.diff()
        vol = daily_volatility(returns, lookback=self.return_vol_lookback, min_periods=self.return_vol_min_periods)

        # Defaulting min periods to half of fast/slow
        fast_min_periods = max(self.fast // 2, 1)
        slow_min_periods = max(self.slow // 2, 1)

        fast_ewma = price.ewm(halflife=halflife(self.fast), min_periods=fast_min_periods).mean()
        slow_ewma = price.ewm(halflife=halflife(self.slow), min_periods=slow_min_periods).mean()

        xover = fast_ewma - slow_ewma
        normalized_xover = xover / vol
        signal = normalized_xover / daily_volatility(normalized_xover, lookback=self.xover_vol_lookback,
                                                     min_periods=self.xover_min_periods)
        return signal
