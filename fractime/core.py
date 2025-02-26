import numpy as np
import polars as pl
import requests
from io import StringIO
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit, float64, int64
from numba.typed import List
from typing import Tuple, List as PyList, Dict
import warnings
import yfinance as yf
import pandas as pd
warnings.filterwarnings('ignore')

def get_yahoo_data(symbol: str, start_date: str) -> pl.DataFrame:
    """Get data from Yahoo Finance using yfinance."""
    try:
        # Get data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date)
        
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")
            
        # Convert pandas DataFrame to Polars
        df = df.reset_index()
        df_pl = pl.from_pandas(df)
        
        # Select only required columns
        df_pl = df_pl.select(['Date', 'Close'])
        
        # Verify the data
        if df_pl.shape[0] == 0:
            raise ValueError("No rows in DataFrame")
            
        return df_pl
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        # Return some sample data for testing
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.now()
        days = (end_date - start_dt).days + 1
        if days <= 0:
            days = 30  # Fallback to 30 days of data if date range is invalid
        dates = [start_dt + timedelta(days=x) for x in range(days)]
        close = np.cumsum(np.random.normal(0, 1, days)) + 100
        
        return pl.DataFrame({
            'Date': dates,
            'Close': close
        })

@njit
def compute_rs(returns: np.ndarray, lag: int) -> float:
    """Compute R/S value for a given lag."""
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0
        
    cumsum = np.cumsum(returns - mean)
    r = np.max(cumsum) - np.min(cumsum)
    return r / std if std > 0 else 0.0

@njit
def linear_regression(x: np.ndarray, y: np.ndarray) -> float:
    """Simple linear regression, returns slope."""
    n = len(x)
    if n < 2:
        return 0.0
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope

@njit
def compute_hurst_exponent(prices: np.ndarray, min_lag: int, max_lag: int) -> float:
    """Optimized Hurst exponent calculation."""
    returns = np.diff(np.log(prices))
    tau = np.empty(max_lag - min_lag)
    rs_values = np.empty(max_lag - min_lag)
    
    for i, lag in enumerate(range(min_lag, max_lag)):
        rs_current = np.empty((len(returns) - lag) // lag)
        
        for j in range(0, len(returns) - lag, lag):
            rs_current[j // lag] = compute_rs(returns[j:j+lag], lag)
            
        valid_rs = rs_current[rs_current > 0]
        if len(valid_rs) > 0:
            tau[i] = np.log(lag)
            rs_values[i] = np.log(np.mean(valid_rs))
        else:
            tau[i] = 0
            rs_values[i] = 0
    
    # Filter out zero values
    mask = (tau > 0) & (rs_values > 0)
    if np.sum(mask) > 1:
        return linear_regression(tau[mask], rs_values[mask])
    return 0.5

@njit
def compute_box_dimension(scaled_prices: np.ndarray, min_window: int, max_window: int, step: int) -> float:
    """Optimized box-counting dimension calculation."""
    dimensions = np.empty((max_window - min_window) // step)
    
    for i, scale in enumerate(range(min_window, max_window, step)):
        boxes = np.ceil(scaled_prices * scale)
        unique_boxes = len(np.unique(boxes))
        dimensions[i] = np.log(unique_boxes) / np.log(scale)
    
    return np.mean(dimensions)

class FractalAnalyzer:
    """Optimized fractal pattern analyzer."""
    
    def __init__(self, min_window: int = 10, max_window: int = 250):
        self.min_window = min_window
        self.max_window = max_window
    
    def analyze_patterns(self, prices: np.ndarray) -> Dict:
        if not isinstance(prices, np.ndarray):
            prices = np.array(prices)
            
        if len(prices.shape) != 1:
            prices = prices.ravel()
            
        if len(prices) < self.min_window:
            raise ValueError(f"Not enough data points. Need at least {self.min_window}, got {len(prices)}")
            
        results = {}
        
        try:
            # Compute Hurst exponent
            results['hurst'] = compute_hurst_exponent(
                prices, 
                self.min_window, 
                self.max_window
            )
            
            # Compute fractal dimension
            # Reshape to 2D array for StandardScaler
            prices_2d = prices.reshape(-1, 1)
            if np.all(prices_2d == prices_2d[0]):
                # If all prices are the same, add tiny noise to avoid scaling issues
                prices_2d = prices_2d + np.random.normal(0, 1e-8, prices_2d.shape)
            
            scaled_prices = StandardScaler().fit_transform(prices_2d).ravel()
            results['fractal_dim'] = compute_box_dimension(
                scaled_prices,
                self.min_window,
                self.max_window,
                20
            )
            
            # Find self-similar patterns
            patterns = self._find_patterns(prices)
            results['self_similar_patterns'] = patterns
        except Exception as e:
            print(f"Error in analyze_patterns: {str(e)}")
            # Return safe default values
            results = {
                'hurst': 0.5,
                'fractal_dim': 1.5,
                'self_similar_patterns': []
            }
            
        return results
    
    @staticmethod
    def _find_patterns(prices: np.ndarray) -> list:
        """Find self-similar patterns in price data."""
        patterns = []
        returns = np.diff(np.log(prices))
        
        for window in range(10, min(250, len(prices)//3)):
            for i in range(len(prices)-window*2):
                pattern1 = prices[i:i+window]
                pattern1_returns = returns[i:i+window-1]
                pattern1_vol = np.std(pattern1_returns)
                pattern2 = prices[i+window:i+window*2]
                pattern2_returns = returns[i+window:i+window*2-1]
                pattern2_vol = np.std(pattern2_returns)
                
                # Skip if either pattern has zero volatility
                if pattern1_vol == 0 or pattern2_vol == 0:
                    continue
                
                # Normalize patterns
                pattern1_norm = (pattern1_returns - np.mean(pattern1_returns)) / pattern1_vol
                pattern2_norm = (pattern2_returns - np.mean(pattern2_returns)) / pattern2_vol
                
                # Compute similarity
                similarity = np.corrcoef(pattern1_norm, pattern2_norm)[0,1]
                
                if similarity > 0.8:  # Only keep strong correlations
                    patterns.append({
                        'start': i,
                        'length': window,
                        'returns': pattern1_returns,
                        'volatility': pattern1_vol,
                        'similarity': similarity,
                        'fractal_dim': compute_box_dimension(
                            StandardScaler().fit_transform(pattern1.reshape(-1, 1)).ravel(),
                            min(5, window//4),
                            min(window, window//2),
                            2
                        )
                    })
        
        return patterns

class FractalSimulator:
    """Generates paths based on fractal patterns and historical distributions."""
    
    def __init__(self, prices: np.ndarray, analyzer: FractalAnalyzer):
        self.prices = prices
        self.analyzer = analyzer
        self.patterns = None
        self.hurst = None
        self._analyze()
    
    def _analyze(self):
        """Perform initial analysis."""
        results = self.analyzer.analyze_patterns(self.prices)
        self.patterns = results['self_similar_patterns']
        self.hurst = results['hurst']
    
    def _compute_volatility_regimes(self, returns: np.ndarray) -> Dict:
        """Compute fractal volatility regimes across multiple scales."""
        # Multiple timeframes for fractal analysis
        timeframes = [5, 21, 63, 126]  # daily, weekly, monthly, quarterly
        
        # Ensure we have enough data
        if len(returns) < max(timeframes):
            # Fall back to smaller timeframes if needed
            timeframes = [tf for tf in timeframes if tf < len(returns)]
            if not timeframes:
                timeframes = [5]  # Minimum timeframe
        
        # Compute volatility at each scale
        vol_states = {}
        for tf in timeframes:
            try:
                # Rolling volatility for this timeframe with safety checks
                rolling_vol = np.array([
                    max(1e-8, np.std(returns[max(0, i-tf):i]))  # Ensure non-zero
                    for i in range(tf, len(returns))
                ])
                
                if len(rolling_vol) < 3:  # Need minimum points for analysis
                    continue
                    
                # Ensure we have valid data before computing
                if np.all(rolling_vol == rolling_vol[0]):  # All values same
                    vol_hurst = 0.5
                    vol_fractal_dim = 1.0
                else:
                    # Compute Hurst exponent of volatility series
                    vol_hurst = compute_hurst_exponent(
                        rolling_vol, 
                        min(5, tf//4),
                        min(50, len(rolling_vol)//3)
                    )
                    
                    # Compute fractal dimension of volatility
                    scaled_vol = StandardScaler().fit_transform(rolling_vol.reshape(-1, 1)).ravel()
                    vol_fractal_dim = compute_box_dimension(
                        scaled_vol,
                        min(5, tf//4),
                        min(tf, len(rolling_vol)//2),
                        2
                    )
                
                # Prepare features for clustering with safety checks
                vol_features = []
                for i in range(len(rolling_vol)-tf):
                    feature_set = [
                        rolling_vol[i],  # Current vol
                        np.log(rolling_vol[i+1] / rolling_vol[i]) if rolling_vol[i] > 0 else 0,  # Vol change
                        max(1e-8, np.std(rolling_vol[i:i+tf]))  # Vol of vol
                    ]
                    vol_features.append(feature_set)
                
                vol_features = np.array(vol_features)
                if len(vol_features) > 0:
                    # Normalize features
                    scaler = StandardScaler()
                    vol_features = scaler.fit_transform(vol_features)
                    
                    # Cluster with minimum 2 clusters if enough data
                    n_clusters = min(3, len(vol_features) // 5)
                    if n_clusters < 2:
                        n_clusters = 2
                    
                    kmeans = KMeans(n_clusters=n_clusters)
                    regime_labels = kmeans.fit_predict(vol_features)
                    regime_centers = kmeans.cluster_centers_
                    
                    vol_states[tf] = {
                        'current': rolling_vol[-1],
                        'history': rolling_vol,
                        'hurst': vol_hurst,
                        'fractal_dim': vol_fractal_dim,
                        'regime_labels': regime_labels,
                        'regime_centers': regime_centers,
                        'current_regime': regime_labels[-1] if len(regime_labels) > 0 else 0
                    }
                    
            except Exception as e:
                print(f"Warning: Error computing volatility regime for timeframe {tf}: {e}")
                continue
        
        # Ensure we have at least one valid state
        if not vol_states:
            # Create a simple fallback state
            vol_states[5] = {
                'current': max(1e-8, np.std(returns[-5:])),
                'history': np.array([max(1e-8, np.std(returns))]),
                'hurst': 0.5,
                'fractal_dim': 1.0,
                'regime_labels': np.array([0]),
                'regime_centers': np.array([[0, 0, 0]]),
                'current_regime': 0
            }
        
        return vol_states

    def simulate_paths(
        self,
        n_steps: int,
        n_paths: int = 1000,
        pattern_weight: float = 0.3,
        cloud_paths: int = 200
    ) -> Tuple[np.ndarray, Dict]:
        """Generate paths using regime-matched sampling based on recent volatility."""
        # Get historical returns
        historical_returns = np.diff(np.log(self.prices))
        
        # Define lookback window as 2x the forecast horizon
        lookback_window = 2 * n_steps
        
        # Get recent volatility regime
        recent_returns = historical_returns[-lookback_window:]
        recent_vol = np.std(recent_returns)
        
        # Find similar volatility regimes in history using multiple metrics
        regime_windows = []
        recent_skew = stats.skew(recent_returns)
        recent_kurt = stats.kurtosis(recent_returns)
        
        for i in range(len(historical_returns) - lookback_window):
            window_returns = historical_returns[i:i+lookback_window]
            window_vol = np.std(window_returns)
            window_skew = stats.skew(window_returns)
            window_kurt = stats.kurtosis(window_returns)
            
            # Calculate similarities using multiple metrics
            vol_similarity = abs(window_vol - recent_vol) / recent_vol
            skew_similarity = abs(window_skew - recent_skew)
            kurt_similarity = abs(window_kurt - recent_kurt)
            
            # Combine similarities with weights
            total_similarity = (0.6 * vol_similarity + 
                              0.25 * skew_similarity + 
                              0.15 * kurt_similarity)
            
            # If combined similarity is good enough, include this window
            if total_similarity < 0.3:  # Start with stricter threshold
                regime_windows.append(i)
        
        # Ensure we have enough similar windows, if not, gradually relax constraint
        similarity_threshold = 0.3
        while len(regime_windows) < 20 and similarity_threshold < 1.0:
            similarity_threshold += 0.1  # More gradual relaxation
            regime_windows = []
            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)
                
                vol_similarity = abs(window_vol - recent_vol) / recent_vol
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)
                
                total_similarity = (0.6 * vol_similarity + 
                                  0.25 * skew_similarity + 
                                  0.15 * kurt_similarity)
                
                if total_similarity < similarity_threshold:
                    regime_windows.append(i)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps))
        
        # Generate paths by sampling from similar regimes
        for i in range(n_paths):
            # Randomly select a similar regime window
            if regime_windows:
                start_idx = np.random.choice(regime_windows)
                # Get n_steps returns from this regime
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                # Randomly select a continuous segment of length n_steps
                segment_start = np.random.randint(0, len(regime_returns) - n_steps)
                path_returns = regime_returns[segment_start:segment_start+n_steps]
            else:
                # Fallback to recent returns if no similar regimes found
                path_returns = np.random.choice(recent_returns, size=n_steps, replace=True)
            
            # Convert returns to price path
            paths[i] = self.prices[-1] * np.exp(np.cumsum(path_returns))
        
        # Cluster and analyze paths
        scaler = StandardScaler()
        scaled_paths = scaler.fit_transform(paths)
        
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(scaled_paths)
        
        # Calculate centroids and probabilities
        centroids = []
        cluster_sizes = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_paths = paths[labels == i]
            centroids.append(np.mean(cluster_paths, axis=0))
            cluster_sizes[i] = len(cluster_paths)
        
        # Compare with patterns across timeframes and compute scores
        cluster_scores = np.zeros(n_clusters)
        pattern_matches = []
        
        # Get patterns for each timeframe
        patterns_by_timeframe = self.patterns  # Now a dict of timeframe -> patterns
        
        for j in range(n_clusters):
            cluster_returns = np.diff(np.log(centroids[j]))
            cluster_prices = centroids[j]
            
            # Get similarities across all timeframes
            similarities = self._compute_path_pattern_similarity(
                cluster_returns,
                cluster_prices,
                patterns_by_timeframe
            )
            
            # Weight the timeframes (more weight to shorter timeframes)
            weighted_sim = (0.5 * similarities.get('daily', 0) +
                          0.3 * similarities.get('weekly', 0) +
                          0.2 * similarities.get('monthly', 0))
            
            cluster_scores[j] = weighted_sim
            
            # Record significant pattern matches
            if weighted_sim > 0.8:
                # Find the best matching pattern from each timeframe
                for tf_name, patterns in patterns_by_timeframe.items():
                    best_match = None
                    best_sim = 0
                    
                    for pattern in patterns:
                        pattern_returns = np.diff(np.log(
                            self.prices[pattern['start']:pattern['start']+pattern['length']]
                        ))
                        
                        if len(cluster_returns) >= len(pattern_returns):
                            sim = self._compute_path_pattern_similarity(
                                cluster_returns[0:len(pattern_returns)],
                                centroids[j][0:len(pattern_returns)+1],
                                {tf_name: [pattern]}
                            )[tf_name]
                            
                            if sim > best_sim:
                                best_sim = sim
                                best_match = pattern
                    
                    if best_match:
                        pattern_matches.append({
                            'cluster': j,
                            'timeframe': tf_name,
                            'pattern_start': best_match['start'],
                            'pattern_length': best_match['length'],
                            'similarity': best_sim
                        })
        
        # Compute final probabilities
        size_probs = cluster_sizes / n_paths
        pattern_weight = 0.4  # Weight for pattern matching vs cluster size
        combined_scores = (1 - pattern_weight) * size_probs + pattern_weight * cluster_scores
        cluster_probs = combined_scores / np.sum(combined_scores)
        
        # Find most likely path
        most_likely_cluster = np.argmax(cluster_probs)
        most_likely_path = centroids[most_likely_cluster]
        
        # Generate probability cloud around most likely path using similar regime sampling
        n_cloud_paths = cloud_paths  # Store the number since cloud_paths will be overwritten
        cloud_paths = np.zeros((n_cloud_paths, n_steps))
        
        # Calculate path probabilities based on regime similarity
        path_probabilities = np.zeros(n_cloud_paths)
        
        for i in range(cloud_paths.shape[0]):
            if regime_windows:
                # Sample from similar regime windows but add more noise
                start_idx = np.random.choice(regime_windows)
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                segment_start = np.random.randint(0, len(regime_returns) - n_steps)
                base_returns = regime_returns[segment_start:segment_start+n_steps]
                
                # Add noise scaled by the regime's volatility
                noise_scale = np.std(regime_returns) * 0.3  # 30% of regime volatility
                cloud_returns = base_returns + np.random.normal(0, noise_scale, size=len(base_returns))
            else:
                # Fallback to sampling from recent returns with noise
                cloud_returns = np.random.choice(recent_returns, size=n_steps, replace=True)
            
            cloud_paths[i] = self.prices[-1] * np.exp(np.cumsum(cloud_returns))
            
            # Calculate probability based on multiple factors:
            # 1. Volatility similarity to recent regime
            path_vol = np.std(cloud_returns)
            vol_similarity = abs(path_vol - recent_vol) / recent_vol
            
            # 2. Return distribution similarity
            path_skew = stats.skew(cloud_returns)
            recent_skew = stats.skew(recent_returns)
            skew_similarity = abs(path_skew - recent_skew)
            
            # Combine similarities with weights
            total_similarity = (0.7 * vol_similarity + 0.3 * skew_similarity)
            path_probabilities[i] = np.exp(-2 * total_similarity)  # Less aggressive decay
        for i, path in enumerate(cloud_paths):
            distance = np.mean(np.abs(path - most_likely_path))
            path_probabilities[i] = np.exp(-distance / np.std(most_likely_path))
        
        # Normalize probabilities
        path_probabilities = path_probabilities / np.sum(path_probabilities)
        
        return paths, {
            'labels': labels,
            'cluster_weights': cluster_scores,
            'cluster_sizes': cluster_sizes,
            'pattern_matches': pattern_matches,
            'centroids': centroids,
            'cluster_probs': cluster_probs,
            'most_likely_path': most_likely_path,
            'probability_cloud': cloud_paths,
            'path_probabilities': path_probabilities
        }

    def _compute_path_pattern_similarity(self, path_returns: np.ndarray, path_prices: np.ndarray, 
                                         patterns_by_timeframe: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Compute similarity between a path and patterns across timeframes."""
        similarities = {}
        
        # Calculate path characteristics
        path_vol = np.std(path_returns)
        path_scaled = StandardScaler().fit_transform(path_prices.reshape(-1, 1)).ravel()
        path_fractal_dim = self.compute_box_dimension(
            path_scaled,
            min(5, len(path_prices)//4),
            min(len(path_prices), len(path_prices)//2),
            2
        )
        
        # Handle patterns as a list
        tf_similarities = []
        
        for pattern in patterns_by_timeframe:
            # Skip if pattern is longer than path
            if pattern['length'] > len(path_returns):
                continue
                
            # Compare fractal dimensions
            fractal_sim = 1 - abs(path_fractal_dim - pattern.get('fractal_dim', 0))
            
            # Compare volatilities
            vol_sim = min(path_vol, pattern.get('volatility', 0)) / max(path_vol, pattern.get('volatility', 1e-8))
            
            # Look for pattern matches within the path
            max_corr = 0.0
            for i in range(len(path_returns) - pattern['length']):
                segment = path_returns[i:i+pattern['length']]
                seg_vol = np.std(segment)
                    
                if seg_vol > 0:
                    # Normalize and compare
                    seg_norm = (segment - np.mean(segment)) / seg_vol
                    pat_norm = (path_returns[pattern['start']:pattern['start']+pattern['length']] - 
                               np.mean(path_returns[pattern['start']:pattern['start']+pattern['length']])) / \
                              pattern['volatility']
                    
                    corr = np.corrcoef(seg_norm, pat_norm)[0,1]
                    max_corr = max(max_corr, corr)
                
                # Combine all similarity metrics
                pattern_sim = (0.5 * max_corr + 
                              0.3 * fractal_sim + 
                              0.2 * vol_sim)
                
                tf_similarities.append(pattern_sim)
            
            # Use the maximum similarity found for this timeframe
            if tf_similarities:
                similarities[tf_name] = max(tf_similarities)
            else:
                similarities[tf_name] = 0.0
        
        return similarities

    @staticmethod
    def compute_box_dimension(data: np.ndarray, min_size: int, max_size: int, step: int) -> float:
        """Compute the box-counting fractal dimension of a time series.
        
        Args:
            data: Input time series
            min_size: Minimum box size
            max_size: Maximum box size
            step: Step size for box scaling
            
        Returns:
            Estimated fractal dimension
        """
        sizes = range(min_size, max_size + 1, step)
        counts = []
        
        for size in sizes:
            # Count number of boxes needed to cover the curve
            boxes = set()
            for i in range(len(data) - 1):
                # Scale to box coordinates
                x = i // size
                y = int(data[i] / (max(data) - min(data)) * size)
                boxes.add((x, y))
            
            counts.append(len(boxes))
        
        # Compute dimension from log-log plot
        log_sizes = np.log([1/s for s in sizes])
        log_counts = np.log(counts)
        
        # Linear regression
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope

    def analyze_path_distributions(self, paths: np.ndarray) -> Dict:
        """Analyze the distribution of simulated paths."""
        n_paths, n_steps = paths.shape
        
        # Calculate return distributions at different horizons
        distributions = {}
        for step in [1, 5, 10, n_steps-1]:  # Different horizons
            returns = np.log(paths[:,step] / paths[:,0])
            distributions[step] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'skew': stats.skew(returns),
                'kurt': stats.kurtosis(returns),
                'quantiles': np.percentile(returns, [1, 25, 50, 75, 99])
            }
        
        # Find most common path shapes
        scaled_paths = StandardScaler().fit_transform(paths)
        kmeans = KMeans(n_clusters=5).fit(scaled_paths)
        
        # Calculate cluster statistics
        clusters = {}
        for i in range(5):
            cluster_paths = paths[kmeans.labels_ == i]
            clusters[i] = {
                'size': len(cluster_paths),
                'mean_path': np.mean(cluster_paths, axis=0),
                'std_path': np.std(cluster_paths, axis=0)
            }
        
        return {
            'distributions': distributions,
            'clusters': clusters
        }

    @staticmethod
    @njit
    def _generate_fbm(n: int, hurst: float) -> np.ndarray:
        """Generate fractional Brownian motion using a simplified method."""
        # Generate Gaussian noise
        noise = np.random.normal(0, 1, n)
        
        # Create time increments
        dt = 1.0 / n
        t = np.arange(n) * dt
        
        # Initialize fBm array
        fbm = np.zeros(n)
        
        # Compute fBm using direct method
        for i in range(1, n):
            # Power-law correlations
            increments = noise[:i] * np.power(t[i] - t[:i], hurst - 0.5)
            fbm[i] = fbm[i-1] + np.sum(increments) * np.sqrt(dt)
        
        # Normalize
        fbm = fbm / np.std(fbm)
        return fbm

class PathAnalyzer:
    """Analyzes and clusters simulation paths."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters)
    
    def analyze_paths(self, paths: np.ndarray) -> Dict:
        """Cluster paths and identify representative trajectories."""
        # Scale paths for clustering - no transpose needed initially
        scaled_paths = self.scaler.fit_transform(paths)
        
        # Perform clustering
        labels = self.kmeans.fit_predict(scaled_paths)
        
        # Find most central path for each cluster
        centroids = self.kmeans.cluster_centers_
        representative_paths = []
        cluster_sizes = []
        
        for i in range(self.n_clusters):
            cluster_paths = paths[labels == i]
            cluster_sizes.append(len(cluster_paths))
            
            # Find path closest to centroid
            distances = np.linalg.norm(
                self.scaler.transform(cluster_paths) - centroids[i],
                axis=1
            )
            representative_paths.append(cluster_paths[np.argmin(distances)])
        
        return {
            'labels': labels,
            'representative_paths': representative_paths,
            'cluster_sizes': cluster_sizes
        }

class FractalVisualizer:
    """Creates interactive visualizations of fractal analysis and simulations."""
    
    @staticmethod
    def plot_analysis_and_forecast(
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """Create comprehensive visualization with return distribution comparison."""
        paths, path_analysis = simulation_results
        
        # Create future dates for forecast
        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=paths.shape[1]+1,
            freq='B'
        )
        
        # Create subplots with proper specifications
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Forecast with Probability Cloud',
                'Return Distribution Comparison',
                'Historical Pattern Matches',
                'Fractal Statistics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],  # First row: price plot and distribution
                [{"type": "xy"}, {"type": "table"}]  # Second row: patterns and stats
            ],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot all paths with probability-based coloring
        cloud_paths = path_analysis['probability_cloud']
        path_probs = path_analysis['path_probabilities']
        
        # Sort paths by probability
        sort_idx = np.argsort(path_probs)
        cloud_paths = cloud_paths[sort_idx]
        path_probs = path_probs[sort_idx]
        
        # Find 95th percentile probability to identify top paths
        prob_95th = np.percentile(path_probs, 95)
        
        # Plot paths from lowest to highest probability
        for path, prob in zip(cloud_paths, path_probs):
            # Aggressive exponential scaling to emphasize only the highest probability paths
            # Scale relative to 95th percentile for better highlighting
            rel_prob = (prob / prob_95th) ** 3  # Cube for more aggressive scaling
            
            if prob > prob_95th:
                # Top 5% paths: Purple to Red transition
                r = 1.0
                b = max(0.0, 1.0 - (prob - prob_95th) / (np.max(path_probs) - prob_95th))
                opacity = 0.8
                width = 2.0
            else:
                # Bottom 95% paths: Light blue with very low opacity
                r = 0.0
                b = 1.0
                opacity = 0.03 + 0.07 * rel_prob  # Very subtle
                width = 0.5
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], path]),
                    name=f'Path (p={prob:.1%})',
                    line=dict(
                        color=f'rgba({int(r*255)},0,{int(b*255)},{opacity})', 
                        width=width
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plot historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Plot most likely path
        most_likely = path_analysis['most_likely_path']
        prob = path_analysis["cluster_probs"][np.argmax(path_analysis["cluster_probs"])]
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], most_likely]),
                name=f'Most Likely Path ({prob:.0%})',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # Pattern matches plot
        if path_analysis['pattern_matches']:
            for match in path_analysis['pattern_matches']:
                start = match['pattern_start']
                cluster = match['cluster']
                sim = match['similarity']
                
                pattern_dates = dates[start:start+20]
                fig.add_trace(
                    go.Scatter(
                        x=pattern_dates,
                        y=historical_prices[start:start+20],
                        name=f'Pattern (sim={sim:.2f})',
                        line=dict(dash='dot')
                    ),
                    row=2, col=1
                )
        
        # Update x-axis formats
        fig.update_xaxes(
            title_text="Date",
            tickformat="%Y-%m-%d",
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            tickformat="%Y-%m-%d",
            row=2, col=1
        )
        
        # Add return distribution comparison
        # Get historical returns for comparison period
        historical_returns = np.diff(np.log(historical_prices[-paths.shape[1]:]))
        
        # Get returns from ALL simulated paths
        all_forecast_returns = np.diff(np.log(paths), axis=1).flatten()
        
        # Plot return distributions
        fig.add_trace(
            go.Histogram(
                x=historical_returns,
                name='Historical Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=all_forecast_returns,
                name='Forecast Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Calculate distribution statistics
        hist_mean = np.mean(historical_returns)
        hist_std = np.std(historical_returns)
        fore_mean = np.mean(all_forecast_returns)
        fore_std = np.std(all_forecast_returns)
        
        # Add distribution statistics annotation
        stats_text = (
            f"Historical: μ={hist_mean:.3%}, σ={hist_std:.3%}<br>"
            f"Forecast: μ={fore_mean:.3%}, σ={fore_std:.3%}"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="x2", yref="y2",
            x=min(min(historical_returns), min(all_forecast_returns)),
            y=0.9,
            showarrow=False,
            font=dict(size=10)
        )
        
        # Update return distribution layout
        fig.update_xaxes(
            title_text="Returns",
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Probability",
            row=1, col=2
        )
        
        # Add explanatory annotation for return distribution
        fig.add_annotation(
            text=(
                "Blue: Historical return distribution<br>"
                "Red: Forecast return distribution<br>"
                "Compare volatility and skewness"
            ),
            xref="x2", yref="y2",
            x=min(min(historical_returns), min(all_forecast_returns)),
            y=0.8,
            showarrow=False,
            font=dict(size=10)
        )
        
        # 3. Statistics table
        stats_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Hurst Exponent', f"{analysis_results['hurst']:.3f}", '>0.5 suggests trend persistence'],
            ['Fractal Dimension', f"{analysis_results['fractal_dim']:.3f}", 'higher = more volatile'],
            ['Pattern Matches', str(len(path_analysis['pattern_matches'])), 'matching forecast paths'],
            ['Most Likely Probability', f"{prob:.1%}", 'based on patterns & clusters'],
            ['Forecast Horizon', f"{paths.shape[1]} steps", 'prediction length']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>', '<b>Interpretation</b>'],
                    font=dict(size=12),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    font=dict(size=11),
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title_text="Fractal Pattern Analysis and Forecast",
            title_x=0.5
        )
        
        # Add explanatory annotations
        fig.add_annotation(
            text="Gray cloud shows possible paths<br>Red line shows most likely trajectory",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            text="Historical patterns matching<br>forecast trajectories",
            xref="paper", yref="paper",
            x=0.02, y=0.45,
            showarrow=False,
            font=dict(size=10)
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Get data
    symbol = "^GSPC"  # S&P 500
    data = get_yahoo_data(symbol, "2020-01-01")
    prices = data['Close'].to_numpy()
    
    # Initialize components
    analyzer = FractalAnalyzer()
    simulator = FractalSimulator(prices, analyzer)
    path_analyzer = PathAnalyzer()
    visualizer = FractalVisualizer()
    
    # Run analysis and simulation
    analysis_results = analyzer.analyze_patterns(prices)
    paths, path_analysis = simulator.simulate_paths(n_steps=30, n_paths=1000)
    path_analysis = path_analyzer.analyze_paths(paths)
    
    # Visualize results
    fig = visualizer.plot_analysis_and_forecast(
        prices,
        (paths, path_analysis),
        analysis_results,
        data['Date'].to_numpy()
    )
    fig.show()