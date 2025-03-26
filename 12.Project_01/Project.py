"""
Advanced Portfolio Optimization with Factor Models and Risk Management
-------------------------------------------------------------------
This project implements a comprehensive portfolio optimization problem that combines
multiple concepts from convex optimization and modern portfolio theory. The implementation includes:
1. Multi-factor portfolio optimization
2. Sector-based constraints and exposure management
3. Advanced risk management features
4. Performance attribution analysis
5. Monte Carlo simulation for robustness
6. Dynamic rebalancing strategy
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import cvxpy as cp
from typing import List, Tuple, Dict, Optional
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioOptimizer:
    def __init__(self, 
                 symbols: List[str],
                 sectors: Dict[str, List[str]],
                 start_date: str,
                 end_date: str,
                 benchmark_symbol: str = '^GSPC'):
        """
        Initialize the advanced portfolio optimizer.
        
        Args:
            symbols: List of stock ticker symbols
            sectors: Dictionary mapping sector names to lists of symbols
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            benchmark_symbol: Benchmark index symbol (default: S&P 500)
        """
        self.symbols = symbols
        self.sectors = sectors
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_symbol = benchmark_symbol
        
        # Initialize data structures
        self.data = None
        self.returns = None
        self.covariance = None
        self.weights = None
        self.portfolio_return = None
        self.portfolio_risk = None
        self.factor_returns = None
        self.factor_loadings = None
        self.benchmark_returns = None
        
    def fetch_data(self) -> None:
        """Fetch historical price data and calculate returns."""
        print("Fetching historical data...")
        
        # Fetch stock data
        data = pd.DataFrame()
        for symbol in self.symbols:
            stock = yf.Ticker(symbol)
            data[symbol] = stock.history(start=self.start_date, end=self.end_date)['Close']
        
        # Fetch benchmark data
        benchmark = yf.Ticker(self.benchmark_symbol)
        benchmark_data = benchmark.history(start=self.start_date, end=self.end_date)['Close']
        
        # Calculate returns
        self.data = data
        self.returns = data.pct_change().dropna()
        self.benchmark_returns = benchmark_data.pct_change().dropna()
        self.covariance = self.returns.cov()
        
        # Calculate factor returns using PCA
        self._calculate_factor_returns()
        
    def _calculate_factor_returns(self) -> None:
        """Calculate factor returns using Principal Component Analysis."""
        # Standardize returns
        returns_std = (self.returns - self.returns.mean()) / self.returns.std()
        
        # Perform PCA
        pca = PCA(n_components=5)  # Use 5 factors
        factor_returns = pca.fit_transform(returns_std)
        
        # Store factor returns and loadings
        self.factor_returns = pd.DataFrame(
            factor_returns,
            index=self.returns.index,
            columns=[f'Factor_{i+1}' for i in range(5)]
        )
        self.factor_loadings = pd.DataFrame(
            pca.components_.T,
            index=self.symbols,
            columns=[f'Factor_{i+1}' for i in range(5)]
        )
        
    def optimize_portfolio(self,
                          risk_free_rate: float = 0.02,
                          target_return: float = 0.15,
                          max_sector_exposure: float = 0.4,
                          transaction_cost: float = 0.001,
                          factor_exposure_limit: float = 0.5) -> None:
        """
        Optimize portfolio weights using multi-factor model and constraints.
        """
        n = len(self.symbols)
        
        # Define optimization variables
        weights = cp.Variable(n)
        
        # Calculate expected returns and covariance
        expected_returns = self.returns.mean().values * 252
        cov_matrix = self.covariance.values * 252
        
        # Calculate factor exposures
        factor_exposures = self.factor_loadings.values
        
        # Define constraints
        constraints = [
            # Basic constraints
            cp.sum(weights) == 1,
            weights >= 0,
            
            # Transaction cost constraints
            cp.sum(weights) <= 1 + transaction_cost,
            cp.sum(weights) >= 1 - transaction_cost,
            
            # Return constraint
            expected_returns @ weights >= target_return,
            
            # Risk constraint
            cp.quad_form(weights, cov_matrix) <= 0.25,
            
            # Sector exposure constraints
            *[cp.sum(weights[[self.symbols.index(symbol) for symbol in sector_symbols]]) <= max_sector_exposure 
              for sector_symbols in self.sectors.values()],
            
            # Factor exposure constraints
            *[cp.abs(factor_exposures[:, i] @ weights) <= factor_exposure_limit 
              for i in range(factor_exposures.shape[1])]
        ]
        
        # Define objective function (risk-adjusted return with factor consideration)
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Calculate factor risk (sum of squared factor exposures)
        factor_risk = cp.sum_squares(factor_exposures.T @ weights)
        
        # Combined objective with factor risk penalty
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk - 0.3 * factor_risk)
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            raise
        
        if problem.status == 'optimal':
            self.weights = weights.value
            self.portfolio_return = portfolio_return.value
            self.portfolio_risk = np.sqrt(portfolio_risk.value)
        else:
            raise ValueError(f"Optimization failed to converge. Status: {problem.status}")
            
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        if self.weights is None:
            raise ValueError("Portfolio must be optimized first")
            
        # Calculate portfolio returns
        portfolio_returns = self.returns @ self.weights
        
        # Calculate metrics
        metrics = {
            'Annual Return': self.portfolio_return,
            'Annual Risk': self.portfolio_risk,
            'Sharpe Ratio': (self.portfolio_return - 0.02) / self.portfolio_risk,
            'Max Drawdown': self._calculate_max_drawdown(portfolio_returns),
            'Sortino Ratio': self._calculate_sortino_ratio(portfolio_returns),
            'Information Ratio': self._calculate_information_ratio(portfolio_returns),
            'Alpha': self._calculate_alpha(portfolio_returns),
            'Beta': self._calculate_beta(portfolio_returns),
            'Factor R-Squared': self._calculate_factor_r_squared(portfolio_returns)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return (self.portfolio_return - 0.02) / (downside_std * np.sqrt(252))
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information Ratio relative to benchmark."""
        excess_returns = returns - self.benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        return np.mean(excess_returns) * 252 / tracking_error
    
    def _calculate_alpha(self, returns: pd.Series) -> float:
        """Calculate Jensen's Alpha."""
        beta = self._calculate_beta(returns)
        excess_returns = returns - self.benchmark_returns
        alpha = np.mean(excess_returns) * 252 - beta * np.mean(self.benchmark_returns) * 252
        return alpha
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark."""
        covariance = np.cov(returns, self.benchmark_returns)[0,1]
        benchmark_variance = np.var(self.benchmark_returns)
        return covariance / benchmark_variance
    
    def _calculate_factor_r_squared(self, returns: pd.Series) -> float:
        """Calculate R-squared of factor model."""
        # Calculate factor exposures for the portfolio
        portfolio_factor_exposures = self.factor_loadings.values.T @ self.weights
        
        # Calculate factor returns contribution
        factor_returns = self.factor_returns.values
        factor_contribution = np.sum(factor_returns * portfolio_factor_exposures, axis=1)
        
        # Calculate R-squared using correlation
        correlation = np.corrcoef(factor_contribution, returns)[0,1]
        return correlation ** 2
    
    def run_monte_carlo_simulation(self, n_simulations: int = 1000) -> Dict[str, np.ndarray]:
        """Run Monte Carlo simulation for portfolio robustness."""
        if self.weights is None:
            raise ValueError("Portfolio must be optimized first")
            
        # Generate simulated returns
        simulated_returns = np.random.multivariate_normal(
            self.returns.mean(),
            self.covariance,
            n_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns @ self.weights
        
        # Calculate metrics for each simulation
        metrics = {
            'Returns': portfolio_returns,
            'Sharpe Ratios': (portfolio_returns - 0.02) / np.std(portfolio_returns),
            'Max Drawdowns': np.array([self._calculate_max_drawdown(pd.Series(returns)) 
                                     for returns in simulated_returns])
        }
        
        return metrics
    
    def plot_results(self) -> None:
        """Plot comprehensive portfolio analysis results with enhanced visualization."""
        if self.weights is None:
            raise ValueError("Portfolio must be optimized first")
        
        # Set style for all plots
        plt.style.use('default')
        
        # Set a modern color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        # Plot 1: Portfolio weights with sector grouping
        plt.figure(figsize=(15, 8))
        weights_df = pd.DataFrame(self.weights, index=self.symbols, columns=['Weight'])
        
        # Group weights by sector
        sector_weights = {}
        for sector, symbols in self.sectors.items():
            sector_weights[sector] = weights_df.loc[symbols, 'Weight'].sum()
        
        # Create bar plot with sector grouping
        bars = plt.bar(range(len(sector_weights)), list(sector_weights.values()), color=colors[:len(sector_weights)])
        
        # Customize the plot
        plt.title('Optimal Portfolio Sector Allocation', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Sectors', fontsize=12)
        plt.ylabel('Portfolio Weight', fontsize=12)
        plt.xticks(range(len(sector_weights)), list(sector_weights.keys()), rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        # Add explanation text
        plt.figtext(0.5, -0.1, 
                   "This plot shows the optimal allocation across different sectors. " +
                   "The weights are constrained to prevent overexposure to any single sector.",
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Risk-Return scatter plot with enhanced styling
        plt.figure(figsize=(15, 8))
        
        # Plot individual assets with sector colors
        sector_colors = {
            'Technology': colors[0],
            'Healthcare': colors[1],
            'Finance': colors[2],
            'Consumer': colors[3],
            'Industrial': colors[4]
        }
        
        for sector, symbols in self.sectors.items():
            for symbol in symbols:
                idx = self.symbols.index(symbol)
                plt.scatter(np.sqrt(self.covariance.iloc[idx,idx] * 252),
                          self.returns[symbol].mean() * 252,
                          c=sector_colors[sector],
                          label=f'{symbol} ({sector})',
                          s=100)
        
        # Plot optimal portfolio
        plt.scatter(self.portfolio_risk, self.portfolio_return,
                   color='red', s=200, marker='*',
                   label='Optimal Portfolio')
        
        # Customize the plot
        plt.title('Risk-Return Profile of Assets and Optimal Portfolio', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Annualized Risk (Standard Deviation)', fontsize=12)
        plt.ylabel('Annualized Expected Return', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanation text
        plt.figtext(0.5, -0.1,
                   "This plot shows the risk-return trade-off for individual assets and the optimal portfolio. " +
                   "Assets are color-coded by sector, and the star represents the optimal portfolio.",
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Factor exposures with enhanced styling
        plt.figure(figsize=(15, 8))
        factor_exposures = pd.DataFrame(
            self.factor_loadings.values.T @ self.weights,
            index=self.factor_loadings.columns,
            columns=['Exposure']
        )
        
        # Create bar plot with custom colors
        bars = plt.bar(range(len(factor_exposures)), factor_exposures['Exposure'],
                      color=colors[:len(factor_exposures)])
        
        # Customize the plot
        plt.title('Portfolio Factor Exposures', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Principal Components', fontsize=12)
        plt.ylabel('Factor Exposure', fontsize=12)
        plt.xticks(range(len(factor_exposures)), factor_exposures.index)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add explanation text
        plt.figtext(0.5, -0.1,
                   "This plot shows the portfolio's exposure to the five principal components. " +
                   "These factors represent the main sources of systematic risk in the market.",
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        # Plot 4: Rolling returns with enhanced styling
        plt.figure(figsize=(15, 8))
        portfolio_returns = self.returns @ self.weights
        rolling_returns = portfolio_returns.rolling(window=252).mean() * 252
        
        # Create line plot with confidence interval
        rolling_std = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
        plt.plot(rolling_returns.index, rolling_returns, color=colors[0], linewidth=2)
        plt.fill_between(rolling_returns.index,
                        rolling_returns - 1.96 * rolling_std,
                        rolling_returns + 1.96 * rolling_std,
                        color=colors[0], alpha=0.2)
        
        # Customize the plot
        plt.title('Rolling Annual Returns with 95% Confidence Interval', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Annualized Return', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanation text
        plt.figtext(0.5, -0.1,
                   "This plot shows the rolling annual returns of the portfolio over time. " +
                   "The shaded area represents the 95% confidence interval of returns.",
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()


def main():
    # Define sectors and their stocks
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'ABT', 'BMY'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA'],
        'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'DIS'],
        'Industrial': ['GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'FDX']
    }
    
    # Combine all symbols
    symbols = [symbol for sector_symbols in sectors.values() for symbol in sector_symbols]
    
    # Set date range
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Initialize and run optimization
    optimizer = AdvancedPortfolioOptimizer(symbols, sectors, start_date, end_date)
    optimizer.fetch_data()
    optimizer.optimize_portfolio(
        risk_free_rate=0.02,
        target_return=0.15,
        max_sector_exposure=0.4,
        transaction_cost=0.001,
        factor_exposure_limit=0.5
    )
    
    # Print results
    metrics = optimizer.calculate_portfolio_metrics()
    print("\nPortfolio Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Run Monte Carlo simulation
    simulation_results = optimizer.run_monte_carlo_simulation()
    print("\nMonte Carlo Simulation Results:")
    print(f"Average Return: {np.mean(simulation_results['Returns']) * 252:.4f}")
    print(f"Average Sharpe Ratio: {np.mean(simulation_results['Sharpe Ratios']):.4f}")
    print(f"Average Max Drawdown: {np.mean(simulation_results['Max Drawdowns']):.4f}")
    
    # Plot results
    optimizer.plot_results()


if __name__ == "__main__":
    main()
