import copy
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import lib as l
from datetime import datetime
import matplotlib.pyplot as plt
import functools
from IPython.display import display, HTML
import importlib

importlib.reload(l)

NUM_DAYS = 252

sns.set_theme(context='talk', style='whitegrid')
plt.rcParams['figure.figsize'] = [30, 10]

class BacktestBase:
    """
    Warning: doesn't gear correctly if we are working across multiple asset classes
    """
    def __init__(self,
                 signals,
                 prices,
                 markets=None,
                 default_lag=2,
                 vol_lookback=50,
                 tilt_lookback=130,
                 uncorrelated_markets=True,
                 lead_lag_range=range(-100, 101),
                 lead_lag_account_curve_range=range(-30, 31, 5),
                 momentum_speeds=[3, 4, 5, 6, 7],
                 plot_long_only=False):
        signals = signals.T.sort_index().T # The (T.sort_index().T) is just ordering all of the columns in alphabetcal order
        prices = prices.T.sort_index().T # The (T.sort_index().T) is just ordering all of the columns in alphabetcal order
        assert(signals.columns.equals(prices.columns))
        
        self.signals = signals
        self.prices = prices
        if markets:
            self.signals = self.signals[markets]
            self.prices = self.prices[markets]
        self.default_lag = default_lag
        self.vol_lookback = vol_lookback
        self.tilt_lookback = tilt_lookback
        self.uncorrelated_markets = uncorrelated_markets
        self.lead_lag_range = lead_lag_range
        self.lead_lag_account_curve_range = lead_lag_account_curve_range
        self.momentum_speeds = momentum_speeds
        self.plot_long_only = plot_long_only

    ###
    # Backtest calculation
    ###
    @functools.cache
    def price_returns(self):
        return self.prices.diff()

    @functools.cache
    def price_returns_vol(self):
        price_returns = self.price_returns()
        return l._daily_volatility(price_returns, lookback=self.vol_lookback)

    @functools.cache
    def price_returns_normalized(self):
        price_returns, price_returns_vol = self.price_returns(), self.price_returns_vol()
        return price_returns / price_returns_vol.shift(1)

    @functools.cache
    def reindexed_signals(self):
        """
        Resample the signals onto the price returns timestamp grid
        """
        price_returns_normalized, signals = self.price_returns_normalized(), self.signals
        return signals.reindex(price_returns_normalized.index, method="ffill")

    @functools.cache
    def market_returns(self, lag=None):
        """
        The backtest returns for each market
        * I dropna if all entries in a row are nan
        * If no lag is specified, default to the default lag
        """
        if lag is None:
            lag = self.default_lag
            
        reindexed_signals, price_returns_normalized = self.reindexed_signals(), self.price_returns_normalized()
        market_returns = reindexed_signals.shift(lag) * price_returns_normalized
        return market_returns.dropna(how="all")

    @functools.cache
    def gearing(self):
        """
        * First we count the number of markets being traded over time. I assumed that once a market starts to be traded it never stops being traded again
        * Then, if the markets are uncorrelated, we gear up by sqrt(num_markets) to counteract div ben. Otherwise gearing is 1
        """
        # Set up the num_markets series as all zeros
        market_returns = self.market_returns()
        market_returns_index = market_returns.index
        num_markets =  pd.Series([0] * len(market_returns_index), index=market_returns_index)
        
        # Find when each market goes live and then increase the num_markets count by from that date forward
        market_start_timestamps = sorted([market_returns[col].first_valid_index() for col in market_returns.columns])
        for timestamp in market_start_timestamps:
            num_markets[timestamp:] += 1

        # If uncorrelated then we gear up by sqrt(num_markets) to counteract div ben. Otherwise we don't do anything
        if self.uncorrelated_markets:
            gearing = np.sqrt(num_markets)
        else:
            gearing = 1

        return gearing
    
    
    # @functools.cache
    def account_curve(self, mask=None, lag=None):
        """
        * Take the mean accross all markets returns and gear up as appropriate
        * I am assuming that I am equally pweighting all of the market returns here so I will need to think this through in future if my use case is that I am not equally weighting them
        * In the end I scale the returns to give 15 vol

        * lag will default to self.lag if lag=None when market_returns is called
        """
        gearing = self.gearing()
        market_returns = self.market_returns(lag=lag)
        if mask is not None:
            market_returns = market_returns[mask]
        market_returns = market_returns.cumsum().ffill().diff() # This inserts zeros wherever there have been nans

        returns = market_returns.mean(axis=1).mul(gearing) # By taking the mean I am assuming that I am equally pweighting all of the market returns here
        normalized_returns = 0.15 * returns / returns.std()
        return normalized_returns.cumsum()

    ###
    # Stats
    ###
    def sharpe(self, lag=None):
        account_curve = self.account_curve(lag=lag)
        returns = account_curve.diff()
        sharpe = l._sharpe(returns)
        return sharpe

    def holding_period(self):
        return l._holding_period(self.reindexed_signals())

    def long_short_account_curves(self):
        signals = self.reindexed_signals()
        long_mask = (signals.shift(self.default_lag) > 0.0)
        short_mask = (signals.shift(self.default_lag) < 0.0)
        
        long_account_curve = self.account_curve(mask=long_mask)
        short_account_curve = self.account_curve(mask=short_mask)

        return long_account_curve, short_account_curve

    def long_short_stats(self):
        long_account_curve, short_account_curve = self.long_short_account_curves()
        long_sharpe = l._sharpe(long_account_curve.diff())
        short_sharpe = l._sharpe(short_account_curve.diff())

        long_returns = long_account_curve.diff()
        short_returns = short_account_curve.diff()

        # This percentage long calculate could probably be done much better
        num_long = len(long_returns[long_returns != 0.0])
        num_short = len(short_returns[short_returns != 0.0])
        
        percentage_long = 100 * round(num_long / (num_long +num_short), 2)
        return pd.Series({"long_sharpe": long_sharpe, "short_sharpe": short_sharpe, "percentage_long": percentage_long}).to_frame("Stats").T
        
    def stats(self):
        stats = pd.Series({"sharpe":  round(self.sharpe(), 2), "holding_period": self.holding_period()}).to_frame("Stats").T
        return stats

    def tilt_timing_stats(self):        
        tilt_signals = self.reindexed_signals().rolling(self.tilt_lookback).mean()
        timing_signals = self.reindexed_signals() - tilt_signals

        tilt_backtest = BacktestBase(tilt_signals, self.prices, default_lag=self.default_lag, uncorrelated_markets = self.uncorrelated_markets)
        timing_backtest = BacktestBase(timing_signals, self.prices, default_lag=self.default_lag, uncorrelated_markets = self.uncorrelated_markets)


        sharpes = pd.Series({"backtest": self.sharpe(), "tilt": tilt_backtest.sharpe(), "timing": timing_backtest.sharpe()}).to_frame("sharpes").T
        account_curves = pd.concat([self.account_curve(), tilt_backtest.account_curve(), timing_backtest.account_curve()], keys=["backtest", "tilt", "timing"], axis=1).diff().dropna(how="any").cumsum()

        return account_curves, sharpes

    def plot_lead_lag(self):
        color = ['Salmon' if i < self.default_lag else 'Skyblue' for i in self.lead_lag_range]
        
        sharpes = {}
        for i in self.lead_lag_range:
            sharpes[i] = self.sharpe(lag=i)
        sharpes = pd.Series(sharpes)
        sharpes.plot.bar(color=color, title="Lead-Lag Plot")
        plt.axhline(0.5, color="black", linestyle="--")
        plt.show()

    def plot_lead_lag_account_curves(self):        
        account_curves = {}
        for i in self.lead_lag_account_curve_range:
            account_curves[i] = self.account_curve(lag=i)
        account_curves = pd.DataFrame(account_curves)
        account_curves = account_curves.diff().dropna(how="any").cumsum()

        max_value = account_curves.columns.max()
        min_value = account_curves.columns.min()
        color_map = {col: str(0.75 - 0.75 * (max_value - col) / (max_value - min_value)) for col in account_curves.columns}

        for col in account_curves.columns:
            plt.plot(account_curves.index, account_curves[col], label=f'Column {col}', color=color_map[col])
        
        plt.legend()
        plt.title('Lead-Lag Account Curves')
        plt.show()

    def plot_tilt_timing(self):
        account_curves, sharpes = self.tilt_timing_stats()
        display(HTML(round(sharpes, 2).to_html()))
        account_curves.plot(title="Tilt vs Timing")
        plt.show()        

    def market_stats(self):
        market_account_curves = self.market_returns().cumsum().ffill()
        market_sharpes = l._sharpe(self.market_returns()).sort_values(ascending=False)
        market_correlations = market_account_curves.diff(5).corr()
        if len(market_correlations) > 1:
            market_correlations = l._cluster_matrix(market_correlations)
        sharpes = market_sharpes.to_frame("Sharpes").T
        return market_account_curves, market_correlations, sharpes

    def turnover(self):
        pass

    def box_plot(self):
        pass

    def drawdown(self):
        pass

class Backtest(BacktestBase):
    def momentum_signals(self):
        mom_signals = {}
        for speed in self.momentum_speeds:
            fast, slow = l.MOM_SPEEDS[speed]
            momentum = l.Momentum(fast, slow)
            mom_signals[speed] = momentum(self.prices)
        mom_signals = pd.concat(mom_signals, axis=1, names=["speed", "markets"])
        mom_signals = mom_signals.T.groupby(level="markets").mean().T
        mom_signals = mom_signals.clip(-2, 2)
        return mom_signals

    def momentum_stats(self):
        momentum_signals = self.momentum_signals()
 
        momentum_backtest = BacktestBase(momentum_signals, self.prices)
        momentum_account_curve = momentum_backtest.account_curve()

        backtest_account_curve = self.account_curve()
        combined_account_curves = pd.concat([backtest_account_curve, momentum_account_curve], keys=["Backtest", "Momentum"], axis=1)
        combined_returns = combined_account_curves.diff().dropna(how="any")
        combined_account_curves = combined_returns.cumsum()

        sharpes = l._sharpe(combined_returns)
        backtest_sharpe = sharpes.Backtest
        momentum_sharpe = sharpes.Momentum
        correlation = combined_account_curves.diff(5).corr().iloc[0, 1]
        appraisal = l._appraisal(backtest_sharpe, momentum_sharpe, correlation)
        stats = pd.DataFrame(dict(backtest_sharpe=backtest_sharpe, momentum_sharpe=momentum_sharpe, correlation=correlation, appraisal=appraisal), index=["Momentum Stats"])

        return combined_account_curves, stats

    def long_only_stats(self):
        long_only_signals = copy.deepcopy(self.signals)
        long_only_signals = long_only_signals.map(lambda x: 1.0)
        long_only_backtest = BacktestBase(long_only_signals, self.prices)
        long_only_account_curve = long_only_backtest.account_curve()

        backtest_account_curve = self.account_curve()
        combined_account_curves = pd.concat([backtest_account_curve, long_only_account_curve], keys=["Backtest", "Long Only"], axis=1)
        combined_returns = combined_account_curves.diff().dropna(how="any")
        combined_account_curves = combined_returns.cumsum()

        sharpes = l._sharpe(combined_returns)
        backtest_sharpe = sharpes["Backtest"]
        long_only_sharpe = sharpes["Long Only"]
        correlation = combined_account_curves.diff(5).corr().iloc[0, 1]
        appraisal = l._appraisal(backtest_sharpe, long_only_sharpe, correlation)
        stats = pd.DataFrame(dict(backtest_sharpe=backtest_sharpe, long_only_sharpe=long_only_sharpe, correlation=correlation, appraisal=appraisal), index=["Long Only Stats"])

        return combined_account_curves, stats
        

    ###
    # Report
    ###
    def report(self):
        l._display("## Stats")
        stats = self.stats()
        display(HTML(stats.to_html()))
        
        l._display("## Account Curve")
        self.account_curve().plot(title="Account Curve")
        plt.show()

        l._display("## Signals")
        self.signals.plot(title="Signals")
        plt.axhline(0.0, color="black", linestyle="--")
        plt.axhline(2.0, color="black")
        plt.axhline(-2.0, color="black")
        plt.show()

        l._display("## Market Stats")
        market_account_curves, _, sharpes = self.market_stats()
        num_markets = len(sharpes.columns)
        fig, ax = plt.subplots(figsize=(num_markets, 1))
        sns.heatmap(round(sharpes, 2), annot=True, cmap='coolwarm', center=0, ax=ax, vmin=-1.0, vmax=1.0, cbar=False)  
        plt.show()
        market_account_curves.plot(title="Market Account Curves")
        plt.show()

        l._display("## Market Correlations")
        _, market_correlations, _ = self.market_stats()
        num_markets = len(market_correlations)
        sns.heatmap(round(market_correlations, 2), annot=True, cmap='coolwarm', center=0, cbar=False)
        plt.show()

        try:
            l._display("## Tilt vs Timing")
            self.plot_tilt_timing()
        except:
            l._display("## Tilt vs Timing")
            print("Error displaying Tilt vs Timing")

        l._display("## Long vs short")
        stats = self.long_short_stats()
        display(HTML(round(stats, 2).to_html()))
        long_account_curve, short_account_curve = self.long_short_account_curves()
        combined = pd.concat([long_account_curve, short_account_curve], keys=["long", "short"], axis=1)
        combined.plot(title="Long vs short account curves")
        plt.show()  

        l._display("## Momentum Comparison")
        combined_account_curves, stats = self.momentum_stats()
        display(HTML(round(stats, 2).to_html()))
        combined_account_curves.plot(title="Momentum Comparison")
        plt.show()

        if self.plot_long_only:
            l._display("## Long Only Comparison")
            combined_account_curves, stats = self.long_only_stats()
            display(HTML(round(stats, 2).to_html()))
            combined_account_curves.plot(title="Long Only Comparison")
            plt.show()

        l._display("## Lead-Lag Plot")
        self.plot_lead_lag()
        
        if self.lead_lag_account_curve_range:
            l._display("## Lead-Lag Account Curves")
            self.plot_lead_lag_account_curves()

    def short_report(self):
        l._display("## Stats")
        stats = self.stats()
        display(HTML(stats.to_html()))
        
        l._display("## Account Curve")
        self.account_curve().plot(title="Account Curve")
        plt.show()

        l._display("## Lead-Lag Plot")
        self.plot_lead_lag()

class Comparison:
    def __init__(self, backtests, plot_lead_lag=True, correlation_lookback=5):
        """
        * backtests here is a dictionary {backtest_name: backtest}
        """
        self.backtests = backtests
        self.correlation_lookback = correlation_lookback
        self.plot_lead_lag = plot_lead_lag

    def __call__(self):
        self.report()

    def account_curves(self):
        account_curves = pd.DataFrame({backtest_name: backtest.account_curve() for backtest_name, backtest in self.backtests.items()})
        return account_curves

    def returns(self):
        returns = self.account_curves().diff()
        return returns

    def correlations(self):
        correlations = self.account_curves().diff(self.correlation_lookback).corr()
        correlations = l._cluster_matrix(correlations)
        correlations = round(correlations, 2)
        return correlations

    def stats(self):
        stats = pd.DataFrame({backtest_name: backtest.stats().squeeze() for backtest_name, backtest in self.backtests.items()})
        return stats

    def momentum_stats(self):
        stats = pd.DataFrame({backtest_name: backtest.momentum_stats()[1].squeeze() for backtest_name, backtest in self.backtests.items()}).T
        return stats


    def sharpes(self):
        sharpes = l._sharpe(self.returns())
        sharpes = round(sharpes, 2)
        sharpes = sharpes.to_frame("Sharpe").T
        return sharpes

    def lead_lags(self):
        sharpes = l._sharpe(self.returns())
        sharpes = round(sharpes, 2)
        sharpes = sharpes.to_frame("Sharpe").T
        return sharpes
        
    def report(self):
        if len(self.backtests) == 1:
            backtest = list(self.backtests.values())[0]
            backtest.report()
        else:
            l._display("## Stats")
            stats = self.stats()
            display(HTML(round(stats, 2).to_html()))
            
            l._display("## Account Curve")
            self.account_curves().plot(title="Account Curve Comparison")
            plt.show()
    
            l._display("## Correlations")
            correlations = self.correlations()
            l._heatmap(correlations.round(2))
            plt.show()

            l._display("## Momentum Comparison")
            stats = self.momentum_stats()
            display(HTML(round(stats, 2).to_html()))
            
            if self.plot_lead_lag:
                l._display("## Lead-Lag Plots")
                for backtest_name, backtest in self.backtests.items():
                    l._ssheader(backtest_name)
                    backtest.plot_lead_lag()
                    plt.show()

def compare(*args, **kwargs):
    comparison = Comparison(*args, **kwargs)
    comparison()