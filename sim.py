import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class FractalPattern:
    """Represents a fractal pattern identified in time series."""
    start_idx: int
    length: int
    hurst: float
    fractal_dim: float
    similarity: float  # Self-similarity score
    next_returns: np.ndarray  # Returns following this pattern
    
@dataclass
class ReturnDistribution:
    """Statistical description of returns following a pattern."""
    mean: float
    std: float
    skew: float
    kurtosis: float
    quantiles: Dict[str, float]
    kde: Optional[stats.gaussian_kde] = None
    mixture_model: Optional[GaussianMixture] = None

class FractalDistributionAnalyzer:
    """Analyzes and models distributions following fractal patterns."""
    
    def __init__(self, 
                window_size: int = 20, 
                pattern_length: int = 40,
                n_regimes: int = 3):
        """
        Initialize the analyzer.
        
        Args:
            window_size: Size of window to use for return distributions
            pattern_length: Length of patterns to identify
            n_regimes: Number of fractal regimes to identify
        """
        self.window_size = window_size
        self.pattern_length = pattern_length
        self.n_regimes = n_regimes
        self.patterns = []
        self.distributions = {}
        self.hmm_model = None
        self.current_regime = None
    
    def extract_fractal_patterns(self, prices: np.ndarray) -> List[FractalPattern]:
        """
        Extract fractal patterns from price time series.
        
        Args:
            prices: Array of prices
            
        Returns:
            List of identified fractal patterns
        """
        patterns = []
        returns = np.diff(np.log(prices))
        
        # Pattern extraction windows
        for i in range(len(prices) - self.pattern_length - self.window_size):
            pattern_prices = prices[i:i+self.pattern_length]
            
            # Calculate fractal properties
            hurst = self._compute_hurst(pattern_prices)
            fractal_dim = 2 - hurst  # Simple relation for brownian time series
            
            # Find similar patterns
            similarity = self._find_pattern_similarity(pattern_prices, prices, i)
            
            # Get future returns
            future_returns = returns[i+self.pattern_length:i+self.pattern_length+self.window_size]
            
            if similarity > 0.6:  # Only keep significant patterns
                patterns.append(FractalPattern(
                    start_idx=i,
                    length=self.pattern_length,
                    hurst=hurst,
                    fractal_dim=fractal_dim,
                    similarity=similarity,
                    next_returns=future_returns
                ))
        
        self.patterns = patterns
        return patterns
    
    def compute_return_distributions(self) -> Dict[str, ReturnDistribution]:
        """
        Compute distributions of returns following each pattern type.
        
        Returns:
            Dictionary mapping pattern types to return distributions
        """
        # Cluster patterns by fractal properties
        if not self.patterns:
            raise ValueError("No patterns extracted. Call extract_fractal_patterns first.")
            
        # Extract features for clustering
        features = np.array([[p.hurst, p.fractal_dim, p.similarity] for p in self.patterns])
        
        # Cluster patterns
        kmeans = KMeans(n_clusters=self.n_regimes)
        labels = kmeans.fit_predict(features)
        
        # Compute distributions for each cluster
        distributions = {}
        for i in range(self.n_regimes):
            cluster_patterns = [p for j, p in enumerate(self.patterns) if labels[j] == i]
            
            if not cluster_patterns:
                continue
                
            # Concatenate all returns for this pattern type
            all_returns = np.concatenate([p.next_returns for p in cluster_patterns])
            
            # Compute distribution statistics
            distribution = ReturnDistribution(
                mean=np.mean(all_returns),
                std=np.std(all_returns),
                skew=stats.skew(all_returns),
                kurtosis=stats.kurtosis(all_returns),
                quantiles={
                    '1%': np.percentile(all_returns, 1),
                    '5%': np.percentile(all_returns, 5),
                    '25%': np.percentile(all_returns, 25),
                    '50%': np.percentile(all_returns, 50),
                    '75%': np.percentile(all_returns, 75),
                    '95%': np.percentile(all_returns, 95),
                    '99%': np.percentile(all_returns, 99)
                }
            )
            
            # Create KDE for sampling
            if len(all_returns) > 10:
                distribution.kde = stats.gaussian_kde(all_returns)
                
                # Fit mixture model if enough data
                if len(all_returns) > 100:
                    try:
                        gm = GaussianMixture(n_components=min(3, len(all_returns)//50), 
                                          random_state=42)
                        gm.fit(all_returns.reshape(-1, 1))
                        distribution.mixture_model = gm
                    except:
                        pass
            
            distributions[f"regime_{i}"] = distribution
        
        self.distributions = distributions
        return distributions
    
    def train_hmm(self) -> hmm.GaussianHMM:
        """
        Train HMM to model transitions between fractal regimes.
        
        Returns:
            Trained HMM model
        """
        if not self.patterns:
            raise ValueError("No patterns extracted. Call extract_fractal_patterns first.")
            
        # Extract features for all data points
        prices_array = np.array([p.start_idx for p in self.patterns])
        
        # Need to ensure we have ordered time points
        if len(prices_array) < 10:
            raise ValueError("Not enough patterns for HMM training")
            
        # Extract features
        features = np.array([[p.hurst, p.fractal_dim, p.similarity] for p in self.patterns])
        
        # Train HMM
        model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=42)
        model.fit(features)
        
        self.hmm_model = model
        return model
    
    def forecast_distribution(self, 
                            current_prices: np.ndarray, 
                            n_samples: int = 1000) -> Tuple[ReturnDistribution, np.ndarray]:
        """
        Forecast return distribution based on current pattern.
        
        Args:
            current_prices: Recent price series
            n_samples: Number of samples to generate
            
        Returns:
            Distribution of forecasted returns and samples
        """
        if not self.hmm_model or not self.distributions:
            raise ValueError("Models not trained. Call train_hmm first.")
            
        # Extract fractal features from current prices
        hurst = self._compute_hurst(current_prices)
        fractal_dim = 2 - hurst
        
        # Find most similar historical pattern
        similarity = self._find_pattern_similarity(current_prices, current_prices, 0)
        
        # Create feature vector
        features = np.array([[hurst, fractal_dim, similarity]])
        
        # Predict regime
        regime_prob = self.hmm_model.predict_proba(features)[0]
        self.current_regime = np.argmax(regime_prob)
        
        # Get distribution for most likely regime
        regime_key = f"regime_{self.current_regime}"
        if regime_key not in self.distributions:
            # Fall back to first regime if not found
            regime_key = list(self.distributions.keys())[0]
        
        distribution = self.distributions[regime_key]
        
        # Generate samples from distribution
        if distribution.mixture_model:
            samples, _ = distribution.mixture_model.sample(n_samples)
            samples = samples.flatten()
        elif distribution.kde:
            samples = distribution.kde.resample(n_samples).flatten()
        else:
            # Fallback to normal distribution
            samples = np.random.normal(
                loc=distribution.mean, 
                scale=distribution.std, 
                size=n_samples
            )
        
        return distribution, samples
    
    def simulate_paths(self, 
                      current_price: float, 
                      n_steps: int, 
                      n_paths: int = 1000) -> np.ndarray:
        """
        Simulate price paths using fractal distribution forecasting.
        
        Args:
            current_price: Current price
            n_steps: Number of steps to simulate
            n_paths: Number of paths to generate
            
        Returns:
            Array of simulated paths
        """
        paths = np.zeros((n_paths, n_steps))
        paths[:,0] = current_price
        
        for i in range(1, n_steps):
            # Extract current pattern for each path
            for j in range(n_paths):
                if i >= self.pattern_length:
                    current_pattern = paths[j, i-self.pattern_length:i]
                else:
                    # Use available data for early steps
                    current_pattern = paths[j, :i]
                
                # Get return distribution
                _, samples = self.forecast_distribution(current_pattern, n_samples=100)
                
                # Sample return and update price
                ret = np.random.choice(samples)
                paths[j,i] = paths[j,i-1] * np.exp(ret)
        
        return paths
    
    def visualize_regimes(self) -> go.Figure:
        """
        Visualize identified regimes and their distributions.
        
        Returns:
            Plotly figure with regime visualization
        """
        if not self.distributions:
            raise ValueError("No distributions computed. Call compute_return_distributions first.")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Return Distributions by Regime", 
                "Regime Transition Probabilities",
                "Fractal Properties by Regime",
                "Forecast Performance"
            )
        )
        
        # Plot return distributions
        x = np.linspace(-0.1, 0.1, 1000)
        for regime, dist in self.distributions.items():
            if dist.kde:
                y = dist.kde(x)
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y, 
                        name=f"{regime} (μ={dist.mean:.3f}, σ={dist.std:.3f})"
                    ),
                    row=1, col=1
                )
                
        # Plot transition matrix if HMM is trained
        if self.hmm_model:
            trans_mat = self.hmm_model.transmat_
            
            fig.add_trace(
                go.Heatmap(
                    z=trans_mat,
                    x=[f"To Regime {i}" for i in range(self.n_regimes)],
                    y=[f"From Regime {i}" for i in range(self.n_regimes)],
                    colorscale="Viridis"
                ),
                row=1, col=2
            )
            
        # Plot fractal properties
        if self.patterns:
            features = np.array([[p.hurst, p.fractal_dim] for p in self.patterns])
            kmeans = KMeans(n_clusters=self.n_regimes)
            labels = kmeans.fit_predict(features)
            
            for i in range(self.n_regimes):
                regime_features = features[labels == i]
                
                if len(regime_features) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_features[:,0],
                            y=regime_features[:,1],
                            mode='markers',
                            name=f"Regime {i}",
                            marker=dict(size=10)
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Fractal Regime Analysis",
            showlegend=True
        )
        
        return fig
    
    def _compute_hurst(self, prices: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis."""
        returns = np.diff(np.log(prices))
        
        # Need a reasonable length for analysis
        if len(returns) < 10:
            return 0.5  # Default value for Brownian motion
            
        # Compute R/S for different lags
        lags = range(2, min(20, len(returns)//2))
        rs_values = []
        
        for lag in lags:
            # Split returns into lag-sized chunks
            chunks = len(returns) // lag
            
            if chunks == 0:
                continue
                
            # Truncate to multiple of lag
            values = returns[:chunks * lag].reshape((chunks, lag))
            
            # Compute means for each chunk
            means = np.mean(values, axis=1)
            
            # De-mean each chunk
            demeaned = values - means.reshape(chunks, 1)
            
            # Compute cumulative sum for each chunk
            cumulative = np.cumsum(demeaned, axis=1)
            
            # Compute R value for each chunk (max - min)
            r_values = np.max(cumulative, axis=1) - np.min(cumulative, axis=1)
            
            # Compute S value for each chunk (standard deviation)
            s_values = np.std(demeaned, axis=1)
            
            # Compute R/S for each chunk
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = r_values / s_values
                rs = rs[~np.isnan(rs)]
                rs = rs[~np.isinf(rs)]
            
            if len(rs) > 0:
                rs_values.append(np.mean(rs))
        
        if len(rs_values) < 2:
            return 0.5
            
        # Get Hurst exponent from log-log regression
        hurst = np.polyfit(np.log(lags[:len(rs_values)]), np.log(rs_values), 1)[0]
        
        # Bound Hurst between 0 and 1
        return max(0.01, min(0.99, hurst))
    
    def _find_pattern_similarity(self, pattern: np.ndarray, series: np.ndarray, start_idx: int) -> float:
        """Find self-similarity of a pattern within the series."""
        pattern_norm = (pattern - np.mean(pattern)) / np.std(pattern)
        
        max_similarity = 0.0
        len_p = len(pattern)
        
        # Look for similar patterns in remaining series
        for i in range(start_idx + len_p, len(series) - len_p):
            window = series[i:i+len_p]
            
            # Normalize window
            window_norm = (window - np.mean(window)) / np.std(window)
            
            # Compute correlation
            corr = np.corrcoef(pattern_norm, window_norm)[0,1]
            max_similarity = max(max_similarity, corr)
        
        return max_similarity

class TradingTimeAnalyzer:
    """Implements Mandelbrot's trading time vs. clock time concept."""
    
    def __init__(self, scaling_factor: float = 0.5):
        self.scaling_factor = scaling_factor
    
    def compute_trading_time(self, 
                           prices: np.ndarray, 
                           volumes: np.ndarray = None) -> np.ndarray:
        """
        Transform clock time to trading time based on volatility and volume.
        
        Args:
            prices: Price time series
            volumes: Optional volume time series
            
        Returns:
            Trading time series
        """
        # Compute returns
        returns = np.diff(np.log(prices))
        
        # Compute local volatility
        window = min(20, len(returns)//5)
        vol = np.array([
            np.std(returns[max(0, i-window):i+1])
            for i in range(len(returns))
        ])
        
        # Normalize volatility
        norm_vol = vol / np.mean(vol)
        
        # Add volume component if available
        if volumes is not None:
            norm_volume = volumes / np.mean(volumes)
            time_increments = (norm_vol * norm_volume) ** self.scaling_factor
        else:
            time_increments = norm_vol ** self.scaling_factor
        
        # Compute cumulative trading time
        trading_time = np.cumsum(time_increments)
        
        # Normalize to [0, 1]
        if len(trading_time) > 0 and trading_time[-1] > 0:
            trading_time = trading_time / trading_time[-1]
            
        return trading_time
    
    def resample_to_trading_time(self, 
                               prices: np.ndarray, 
                               trading_time: np.ndarray, 
                               n_points: int = 100) -> np.ndarray:
        """
        Resample prices to uniform trading time.
        
        Args:
            prices: Price time series
            trading_time: Trading time series
            n_points: Number of points in resampled series
            
        Returns:
            Resampled price series
        """
        # Generate uniform trading time points
        uniform_time = np.linspace(0, 1, n_points)
        
        # Resample prices to uniform trading time
        resampled_prices = np.interp(uniform_time, trading_time, prices)
        
        return resampled_prices

class ScalingAnalyzer:
    """Analyzes scaling properties and self-similarity across different time scales."""
    
    def __init__(self, min_scale: int = 5, max_scale: int = None):
        """
        Initialize the scaling analyzer.
        
        Args:
            min_scale: Minimum scale window size to analyze
            max_scale: Maximum scale window size to analyze
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scaling_exponents = {}
        self.scale_similarities = {}
    
    def analyze_scaling(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Analyze scaling properties of the time series.
        
        Args:
            prices: Price time series
            
        Returns:
            Dictionary of scaling metrics
        """
        returns = np.diff(np.log(prices))
        
        max_scale = self.max_scale or len(returns) // 4
        scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(max_scale), 
            num=10, 
            dtype=int
        )
        scales = np.unique(scales)  # Remove duplicates
        
        # Compute volatility at different scales
        vol_by_scale = {}
        for scale in scales:
            # Skip scales that are too large
            if scale >= len(returns):
                continue
                
            # Compute returns at this scale
            scale_returns = np.array([
                np.sum(returns[i:i+scale])
                for i in range(0, len(returns)-scale, scale)
            ])
            
            # Store volatility
            vol_by_scale[scale] = np.std(scale_returns)
        
        # Compute scaling exponent (should be 0.5 for random walk)
        scales_list = np.array(list(vol_by_scale.keys()))
        vols_list = np.array(list(vol_by_scale.values()))
        
        if len(scales_list) > 1:
            # H in vol ~ t^H relation
            H, _ = np.polyfit(np.log(scales_list), np.log(vols_list), 1)
            scaling_exponent = H / 2  # Convert to Hurst
        else:
            scaling_exponent = 0.5
        
        self.scaling_exponents = {
            'H': scaling_exponent,
            'scales': scales_list,
            'volatilities': vols_list
        }
        
        return self.scaling_exponents
    
    def compute_self_similarity(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute self-similarity across different time scales.
        
        Args:
            prices: Price time series
            
        Returns:
            Dictionary of self-similarity metrics
        """
        # Set up scales for comparison
        max_scale = self.max_scale or len(prices) // 10
        scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(max_scale), 
            num=5, 
            dtype=int
        )
        scales = np.unique(scales)
        
        similarities = {}
        
        # Downsample to different scales
        for scale in scales:
            if scale >= len(prices) // 3:
                continue
                
            # Downsample by averaging
            downsampled = np.array([
                np.mean(prices[i:i+scale])
                for i in range(0, len(prices)-scale, scale)
            ])
            
            # Normalize both series for comparison
            norm_orig = self._normalize_series(prices[:len(downsampled)*scale:scale])
            norm_down = self._normalize_series(downsampled)
            
            # Compute similarity
            similarity = np.corrcoef(norm_orig, norm_down)[0, 1]
            similarities[scale] = similarity
        
        self.scale_similarities = similarities
        return similarities
    
    def visualize_scaling(self) -> go.Figure:
        """
        Visualize scaling properties of the time series.
        
        Returns:
            Plotly figure with scaling visualization
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Volatility Scaling", "Self-Similarity Across Scales")
        )
        
        # Plot volatility scaling
        if self.scaling_exponents:
            scales = self.scaling_exponents['scales']
            vols = self.scaling_exponents['volatilities']
            H = self.scaling_exponents['H']
            
            # Actual data
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=vols,
                    mode='markers',
                    name='Observed Volatility',
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
            
            # Fitted line
            x_fit = np.linspace(min(scales), max(scales), 100)
            y_fit = np.exp(np.log(vols[0]) + H * (np.log(x_fit) - np.log(scales[0])))
            
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'Fitted (H={H:.3f})',
                    line=dict(dash='dash')
                ),
                row=1, col=1
            )
            
            # Brownian reference
            y_brownian = np.exp(np.log(vols[0]) + 0.5 * (np.log(x_fit) - np.log(scales[0])))
            
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_brownian,
                    mode='lines',
                    name='Brownian (H=0.5)',
                    line=dict(dash='dot')
                ),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text="Time Scale", type="log", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", type="log", row=1, col=1)
        
        # Plot self-similarity
        if self.scale_similarities:
            scales = list(self.scale_similarities.keys())
            similarities = list(self.scale_similarities.values())
            
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=similarities,
                    mode='lines+markers',
                    name='Self-Similarity'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Time Scale", type="log", row=1, col=2)
            fig.update_yaxes(title_text="Similarity", row=1, col=2)
        
        fig.update_layout(
            height=500,
            width=1000,
            title_text="Scaling Analysis"
        )
        
        return fig
    
    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """Normalize a time series for comparison."""
        return (series - np.mean(series)) / np.std(series)

# Example usage functions
def demo_fractal_distribution_analyzer(prices: np.ndarray) -> FractalDistributionAnalyzer:
    """Run a demo of the FractalDistributionAnalyzer."""
    # Initialize analyzer
    analyzer = FractalDistributionAnalyzer(
        window_size=20,
        pattern_length=40,
        n_regimes=3
    )
    
    # Extract patterns
    patterns = analyzer.extract_fractal_patterns(prices)
    print(f"Found {len(patterns)} fractal patterns")
    
    # Compute distributions
    distributions = analyzer.compute_return_distributions()
    for regime, dist in distributions.items():
        print(f"{regime}: μ={dist.mean:.6f}, σ={dist.std:.6f}, skew={dist.skew:.3f}")
    
    # Train HMM
    try:
        analyzer.train_hmm()
        print("HMM trained successfully")
    except Exception as e:
        print(f"HMM training error: {e}")
    
    return analyzer

def demo_trading_time(prices: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
    """Run a demo of the trading time concept."""
    time_analyzer = TradingTimeAnalyzer()
    trading_time = time_analyzer.compute_trading_time(prices, volumes)
    
    # Resample to uniform trading time
    resampled = time_analyzer.resample_to_trading_time(prices, trading_time)
    
    print(f"Original length: {len(prices)}, Resampled length: {len(resampled)}")
    return resampled 

def demo_scaling_analysis(prices: np.ndarray) -> ScalingAnalyzer:
    """Run a demo of the scaling analysis."""
    analyzer = ScalingAnalyzer()
    
    # Analyze scaling properties
    scaling = analyzer.analyze_scaling(prices)
    print(f"Scaling exponent H = {scaling['H']:.3f}")
    
    # Compute self-similarity
    similarities = analyzer.compute_self_similarity(prices)
    for scale, sim in similarities.items():
        print(f"Self-similarity at scale {scale}: {sim:.3f}")
    
    return analyzer 