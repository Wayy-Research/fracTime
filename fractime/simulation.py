"""
Path simulation for fractal time series.

This module provides simulation engines for generating price paths
based on fractal patterns and historical distributions.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from numba import njit

from .utils import _ensure_numpy_array, compute_hurst_exponent, compute_box_dimension
from .analysis import FractalAnalyzer, CrossDimensionalAnalyzer


class TradingTimeWarper:
    """
    Transforms between clock time and trading time based on volatility regimes.

    This class implements Mandelbrot's concept that markets operate on their own
    time scale that flows faster during high volatility and slower during low volatility.
    """

    def __init__(self, alpha: float = 0.5, min_scale: float = 0.1, max_scale: float = 10.0):
        """
        Initialize trading time warper.

        Args:
            alpha: Scaling factor that controls transformation intensity (0.5 default)
            min_scale: Minimum scaling factor to prevent extreme compression (0.1 default)
            max_scale: Maximum scaling factor to prevent extreme expansion (10.0 default)
        """
        self.alpha = alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.time_map = None
        self.inverse_map = None

    def compute_time_transformation(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Compute transformation between clock time and trading time.

        Args:
            prices: Historical price series
            volumes: Optional volume series (if available)

        Returns:
            Dictionary with time mapping information
        """
        # Convert to NumPy arrays if needed (e.g., from Polars Series)
        prices = _ensure_numpy_array(prices)
        volumes = _ensure_numpy_array(volumes)

        n = len(prices)

        # Calculate returns and volatility
        log_returns = np.diff(np.log(prices))

        # Use rolling windows to estimate local volatility
        window_sizes = [5, 21, 63]  # Day, week, month in trading days
        volatility_series = {}

        for window in window_sizes:
            # Compute rolling volatility with window
            vol = np.zeros(n)
            for i in range(window, n):
                vol[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)  # Annualized

            # Fill initial points with first valid value
            # Handle edge case where window >= n
            if window < n:
                vol[:window] = vol[window]
            else:
                # If window is >= data length, compute volatility over all available data
                vol[:] = np.std(log_returns) * np.sqrt(252)
            volatility_series[window] = vol

        # Combine volatilities with different weights (emphasize recent)
        weights = {5: 0.6, 21: 0.3, 63: 0.1}
        combined_vol = np.zeros(n)

        for window, vol in volatility_series.items():
            combined_vol += weights[window] * vol

        # Normalize volatility to have mean 1
        relative_vol = combined_vol / np.mean(combined_vol)

        # If volumes are provided, incorporate them
        if volumes is not None and len(volumes) == n:
            # Normalize volumes to have mean 1
            relative_vol_normalized = volumes / np.mean(volumes)
            # Combined activity measure (volatility and volume)
            activity = np.sqrt(relative_vol * relative_vol_normalized)
        else:
            # Just use volatility if volume not available
            activity = relative_vol

        # Apply power transformation with bounds
        time_dilation = np.power(activity, self.alpha)
        time_dilation = np.clip(time_dilation, self.min_scale, self.max_scale)

        # Compute cumulative trading time
        trading_time = np.cumsum(time_dilation)

        # Normalize to span same overall duration
        trading_time = trading_time * (n - 1) / trading_time[-1]

        # Store the mapping
        self.time_map = {
            'clock_to_trading': trading_time,
            'trading_time_values': trading_time,
            'clock_time_indices': np.arange(n),
            'dilation_factors': time_dilation,
            'activity': activity
        }

        return self.time_map

    def transform_to_trading_time(self, series: np.ndarray) -> np.ndarray:
        """
        Transform a time series from clock time to trading time.

        Args:
            series: Time series in clock time

        Returns:
            Time series resampled to trading time
        """
        if self.time_map is None:
            raise ValueError("Must call compute_time_transformation first")

        # Get original points
        x_orig = np.arange(len(series))
        y_orig = series

        # Get trading time points
        x_trading = self.time_map['trading_time_values']

        # Create uniform grid in trading time
        x_uniform = np.linspace(0, x_trading[-1], len(series))

        # Resample onto uniform trading time grid
        y_trading = np.interp(x_uniform, x_trading, y_orig)

        return y_trading

    def transform_path_to_trading_time(self, path: np.ndarray) -> np.ndarray:
        """
        Transform a path from clock time to trading time.

        Args:
            path: Path in clock time

        Returns:
            Path resampled to trading time
        """
        return self.transform_to_trading_time(path)

    def transform_paths_to_trading_time(self, paths: np.ndarray) -> np.ndarray:
        """
        Transform multiple paths from clock time to trading time.

        Args:
            paths: Array of paths in clock time (n_paths, n_steps)

        Returns:
            Array of paths in trading time
        """
        warped_paths = np.zeros_like(paths)

        for i in range(paths.shape[0]):
            warped_paths[i] = self.transform_path_to_trading_time(paths[i])

        return warped_paths

    def transform_to_clock_time(self, trading_series: np.ndarray) -> np.ndarray:
        """
        Transform a time series from trading time to clock time.

        Args:
            trading_series: Time series in trading time

        Returns:
            Time series resampled to clock time
        """
        if self.time_map is None:
            raise ValueError("Must call compute_time_transformation first")

        # Create uniform trading time grid (assuming trading_series is on uniform grid)
        x_trading_uniform = np.linspace(0, self.time_map['trading_time_values'][-1], len(trading_series))

        # Get original time points
        x_clock = np.arange(len(self.time_map['trading_time_values']))
        x_trading = self.time_map['trading_time_values']

        # Resample from uniform trading time back to clock time
        resampled = np.interp(x_trading, x_trading_uniform, trading_series)

        return resampled


class PathAnalyzer:
    """Analyzes and clusters simulation paths."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters)

    def analyze_paths(self, paths: np.ndarray) -> Dict:
        """Cluster paths and identify representative trajectories."""
        # Scale paths for clustering
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


class FractalSimulator:
    """Generates paths based on fractal patterns and historical distributions."""

    def __init__(self, prices: np.ndarray, analyzer: FractalAnalyzer, volumes: np.ndarray = None):
        # Convert to NumPy arrays if needed (e.g., from Polars Series)
        self.prices = _ensure_numpy_array(prices)
        self.analyzer = analyzer
        self.patterns = None
        self.hurst = None
        self.volumes = _ensure_numpy_array(volumes)
        self.time_warper = TradingTimeWarper()
        self.cross_dim_analyzer = None
        self.quantum_generator = None
        self.quantum_levels = None
        self._analyze()

        # Set up trading time warping if we have volumes
        if self.volumes is not None and len(self.volumes) == len(self.prices):
            self.time_map = self.time_warper.compute_time_transformation(self.prices, self.volumes)
            print("Trading time mapping computed with price and volume data")

            # Initialize cross-dimensional analyzer if we have volumes
            self.cross_dim_analyzer = CrossDimensionalAnalyzer()
            self.cross_dim_analyzer.add_dimension("price", self.prices)
            self.cross_dim_analyzer.add_dimension("volume", self.volumes)

            # Perform initial cross-dimensional analysis
            self.cross_dim_results = self.cross_dim_analyzer.analyze_dimensions()
            print("Cross-dimensional fractal analysis completed")
        else:
            # Just use prices for time warping if no volumes
            self.time_map = self.time_warper.compute_time_transformation(self.prices)
            print("Trading time mapping computed with price data only")

        # Initialize quantum price level generator (optional feature)
        try:
            from fractime.quantum import QuantumPriceLevelGenerator
            self.quantum_generator = QuantumPriceLevelGenerator(energy_levels=5)
            self.quantum_levels = self.quantum_generator.generate_price_levels(self.prices)
            print("Quantum price levels generated")
        except ImportError:
            print("Quantum module not available, skipping quantum price levels")
            self.quantum_generator = None
            self.quantum_levels = None

        # Prepare sampled data for faster simulations
        if len(self.prices) > 1000:
            # Create downsampled version for faster regime matching
            sampling_rate = len(self.prices) // 1000
            self.sampled_prices = self.prices[::sampling_rate]
            if self.volumes is not None:
                self.sampled_volumes = self.volumes[::sampling_rate]
            print(f"Created downsampled data: {len(self.sampled_prices)} points")
        else:
            self.sampled_prices = self.prices
            if self.volumes is not None:
                self.sampled_volumes = self.volumes

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
        cloud_paths: int = 200,
        preserve_volatility: bool = True,
        use_trading_time: bool = True,  # Parameter for trading time warping
        warping_alpha: float = 0.5,     # Parameter for warping intensity
        enable_time_forecast: bool = True,  # Whether to forecast time dilation into the future
        use_cross_dim: bool = True,  # Parameter for cross-dimensional filtering
        use_quantum_levels: bool = True,  # Parameter for quantum price level filtering
        quantum_influence: float = 0.5  # How strongly quantum levels influence path selection (0-1)
    ) -> Tuple[np.ndarray, Dict]:
        """Generate paths using regime-matched sampling based on trading time transformation."""
        # Get historical returns
        historical_returns = np.diff(np.log(self.prices))

        # Define lookback window as 2x the forecast horizon
        lookback_window = 2 * n_steps

        # Get recent volatility regime
        recent_returns = historical_returns[-lookback_window:]
        recent_vol = np.std(recent_returns)

        # Update time warper settings if changed
        if warping_alpha != self.time_warper.alpha:
            self.time_warper.alpha = warping_alpha
            # Recompute time warping
            if self.volumes is not None and len(self.volumes) == len(self.prices):
                self.time_map = self.time_warper.compute_time_transformation(self.prices, self.volumes)
            else:
                self.time_map = self.time_warper.compute_time_transformation(self.prices)

        # Find similar volatility regimes in history using multiple metrics
        regime_windows = []
        recent_skew = stats.skew(recent_returns)
        recent_kurt = stats.kurtosis(recent_returns)

        # If we're using trading time, incorporate dilation factors into regime matching
        if use_trading_time:
            # Get recent time dilation factors
            recent_dilation = self.time_map['dilation_factors'][-lookback_window:]
            recent_dilation_mean = np.mean(recent_dilation)
            recent_dilation_std = np.std(recent_dilation)

            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)

                # Get time dilation for this window
                window_dilation = self.time_map['dilation_factors'][i:i+lookback_window]
                window_dilation_mean = np.mean(window_dilation)
                window_dilation_std = np.std(window_dilation)

                # Calculate similarities using multiple metrics
                vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)

                # Time dilation similarity (both mean level and variability)
                dilation_mean_sim = abs(window_dilation_mean - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)
                dilation_std_sim = abs(window_dilation_std - recent_dilation_std) / max(recent_dilation_std, 1e-8)

                # Combine similarities with weights (add time dilation factors)
                total_similarity = (
                    0.4 * vol_similarity +
                    0.15 * skew_similarity +
                    0.15 * kurt_similarity +
                    0.2 * dilation_mean_sim +
                    0.1 * dilation_std_sim
                )

                # If combined similarity is good enough, include this window
                if total_similarity < 0.3:  # Start with stricter threshold
                    regime_windows.append(i)
        else:
            # Original regime matching without time warping
            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)

                # Calculate similarities using multiple metrics
                vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)

                # Combine similarities with weights
                total_similarity = (
                    0.6 * vol_similarity +
                    0.25 * skew_similarity +
                    0.15 * kurt_similarity
                )

                # If combined similarity is good enough, include this window
                if total_similarity < 0.3:  # Start with stricter threshold
                    regime_windows.append(i)

        # Ensure we have enough similar windows, if not, gradually relax constraint
        similarity_threshold = 0.3
        while len(regime_windows) < 20 and similarity_threshold < 1.0:
            similarity_threshold += 0.1  # More gradual relaxation
            regime_windows = []

            if use_trading_time:
                # With time warping
                for i in range(len(historical_returns) - lookback_window):
                    window_returns = historical_returns[i:i+lookback_window]
                    window_vol = np.std(window_returns)
                    window_skew = stats.skew(window_returns)
                    window_kurt = stats.kurtosis(window_returns)

                    # Get time dilation for this window
                    window_dilation = self.time_map['dilation_factors'][i:i+lookback_window]
                    window_dilation_mean = np.mean(window_dilation)
                    window_dilation_std = np.std(window_dilation)

                    # Calculate similarities
                    vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                    skew_similarity = abs(window_skew - recent_skew)
                    kurt_similarity = abs(window_kurt - recent_kurt)
                    dilation_mean_sim = abs(window_dilation_mean - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)
                    dilation_std_sim = abs(window_dilation_std - recent_dilation_std) / max(recent_dilation_std, 1e-8)

                    # Combined similarity
                    total_similarity = (
                        0.4 * vol_similarity +
                        0.15 * skew_similarity +
                        0.15 * kurt_similarity +
                        0.2 * dilation_mean_sim +
                        0.1 * dilation_std_sim
                    )

                    if total_similarity < similarity_threshold:
                        regime_windows.append(i)
            else:
                # Original approach
                for i in range(len(historical_returns) - lookback_window):
                    window_returns = historical_returns[i:i+lookback_window]
                    window_vol = np.std(window_returns)
                    window_skew = stats.skew(window_returns)
                    window_kurt = stats.kurtosis(window_returns)

                    vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                    skew_similarity = abs(window_skew - recent_skew)
                    kurt_similarity = abs(window_kurt - recent_kurt)

                    total_similarity = (
                        0.6 * vol_similarity +
                        0.25 * skew_similarity +
                        0.15 * kurt_similarity
                    )

                    if total_similarity < similarity_threshold:
                        regime_windows.append(i)

        # Initialize paths array
        paths = np.zeros((n_paths, n_steps))

        # For trading time approach, store dilation factors for each path
        if use_trading_time and enable_time_forecast:
            path_dilation_factors = np.zeros((n_paths, n_steps))

        # Generate paths by sampling from similar regimes
        for i in range(n_paths):
            # Randomly select a similar regime window
            if regime_windows:
                start_idx = np.random.choice(regime_windows)
                # Get n_steps returns from this regime
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                # Randomly select a continuous segment of length n_steps
                # Handle edge case where regime_returns length equals n_steps
                max_start = max(0, len(regime_returns) - n_steps)
                if max_start == 0:
                    segment_start = 0
                else:
                    segment_start = np.random.randint(0, max_start)
                path_returns = regime_returns[segment_start:segment_start+n_steps]

                # For trading time approach, get corresponding dilation factors
                if use_trading_time and enable_time_forecast:
                    dilation_start = start_idx + segment_start
                    dilation_end = dilation_start + n_steps
                    if dilation_end <= len(self.time_map['dilation_factors']):
                        path_dilation_factors[i] = self.time_map['dilation_factors'][dilation_start:dilation_end]
                    else:
                        # Use repeating recent values if we go beyond available factors
                        recent_dilation = self.time_map['dilation_factors'][-lookback_window//2:]
                        path_dilation_factors[i] = np.random.choice(recent_dilation, size=n_steps)
            else:
                # Fallback to recent returns if no similar regimes found
                path_returns = np.random.choice(recent_returns, size=n_steps, replace=True)

                # For trading time approach, fallback to recent dilation factors
                if use_trading_time and enable_time_forecast:
                    recent_dilation = self.time_map['dilation_factors'][-lookback_window//2:]
                    path_dilation_factors[i] = np.random.choice(recent_dilation, size=n_steps)

            # Convert returns to price path
            paths[i] = self.prices[-1] * np.exp(np.cumsum(path_returns))

        # If using trading time, resample paths to account for time dilation
        if use_trading_time and enable_time_forecast:
            # Apply trading time transformations to each path
            warped_paths = np.zeros_like(paths)

            for i in range(n_paths):
                # Create a custom time warper for this path's forecasted time dilation
                path_warper = TradingTimeWarper(alpha=self.time_warper.alpha)

                # Create a simple mapping of clock time to trading time for this path
                dilation_factors = path_dilation_factors[i]
                trading_time = np.cumsum(dilation_factors)

                # Normalize to same total duration
                trading_time = trading_time * (n_steps - 1) / trading_time[-1]

                # Create a time map just for this path
                path_time_map = {
                    'trading_time_values': trading_time,
                    'clock_time_indices': np.arange(n_steps),
                    'dilation_factors': dilation_factors
                }

                # Use our custom transformation on the path
                x_orig = np.arange(n_steps)
                y_orig = paths[i]

                # Create uniform grid in trading time
                x_uniform = np.linspace(0, trading_time[-1], n_steps)

                # Resample onto uniform trading time grid (essentially stretching/compressing time)
                warped_paths[i] = np.interp(x_uniform, trading_time, y_orig)

            # Replace original paths with time-warped paths
            paths = warped_paths

        # Apply cross-dimensional filtering if enabled and we have volume data
        cross_dim_weights = None
        cross_dim_regime = None

        if use_cross_dim and self.cross_dim_analyzer is not None and self.volumes is not None:
            # We'll score each path based on how well it matches the cross-dimensional
            # fractal properties of the historical data

            cross_dim_weights = np.ones(n_paths)

            # Get current regime information
            cross_dim_regime = self.cross_dim_results.get('regime', {})
            current_regime = cross_dim_regime.get('regime', 0)
            regime_confidence = cross_dim_regime.get('confidence', 0.5)

            # Get cross correlation between price and volume
            price_vol_corr = self.cross_dim_results.get('cross_correlation', [[1, 0], [0, 1]])[0][1]

            # Perform analysis on each path
            for i in range(n_paths):
                path = paths[i]

                # Create a fictional volume path that preserves the cross correlation
                if hasattr(self, 'sampled_volumes') and self.sampled_volumes is not None:
                    # Create a volume series that correlates with this path's price
                    # at approximately the same level as in the historical data

                    # Compute path returns
                    path_returns = np.diff(np.log(path))

                    # Measure price/volume correlation for this path
                    # Check if we can use the simulated volume from the same regime
                    if regime_windows and i < len(regime_windows):
                        # Get historical volume data from this regime
                        start_idx = regime_windows[i % len(regime_windows)]
                        if start_idx + n_steps < len(self.volumes):
                            # Extract a segment of historical volume
                            sim_volume = self.volumes[start_idx:start_idx+n_steps-1]

                            # Calculate log changes
                            vol_changes = np.diff(np.log(sim_volume + 1))  # Add 1 to avoid log(0)

                            # Calculate correlation
                            if len(vol_changes) > 1 and len(path_returns) > 1:
                                try:
                                    # Calculate correlation between price and volume paths
                                    path_corr = np.corrcoef(path_returns, vol_changes)[0, 1]

                                    # Calculate correlation similarity score
                                    # Compare to historical price-volume correlation
                                    corr_similarity = 1.0 - min(1.0, abs(path_corr - price_vol_corr))

                                    # Weight paths by similarity to historical correlation patterns
                                    # Higher weights for paths that preserve the price-volume relationship
                                    cross_dim_weights[i] = 0.2 + 0.8 * corr_similarity
                                except:
                                    # Keep default weight on error
                                    pass

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

        # Find similar patterns in historical data
        patterns = self.analyzer.get_patterns(self.prices)

        # Get pattern similarity - fix the call to match our updated method
        try:
            # Check what type of patterns we're working with
            if isinstance(patterns, dict):
                # Handle dictionary of patterns by timeframe (original structure)
                similarities_by_timeframe = {}
                for timeframe, timeframe_patterns in patterns.items():
                    # Convert each timeframe's patterns to array form if needed
                    pattern_arrays = []
                    for pattern in timeframe_patterns:
                        if 'start' in pattern and 'length' in pattern:
                            # Extract the actual price segment for this pattern
                            start = pattern['start']
                            length = pattern['length']
                            if start + length <= len(self.prices):
                                pattern_arrays.append(self.prices[start:start+length])

                    # Compute similarity for this timeframe's patterns
                    if pattern_arrays:
                        similarities_by_timeframe[timeframe] = np.mean(
                            self._compute_path_pattern_similarity(self.prices, pattern_arrays)
                        )
                    else:
                        similarities_by_timeframe[timeframe] = 0.0

                # Calculate weighted average across timeframes (original logic)
                weighted_similarity = (0.5 * similarities_by_timeframe.get('daily', 0) +
                                     0.3 * similarities_by_timeframe.get('weekly', 0) +
                                     0.2 * similarities_by_timeframe.get('monthly', 0))
            else:
                # Handle direct list of pattern arrays (new structure)
                similarities = self._compute_path_pattern_similarity(self.prices, patterns)
                weighted_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        except Exception as e:
            print(f"Error computing pattern similarity: {e}")
            weighted_similarity = 0.0  # Fallback to zero similarity

        cluster_scores = weighted_similarity * np.ones(n_clusters)

        # Calculate final probabilities
        size_probs = cluster_sizes / n_paths

        # Start with standard pattern-based scoring
        combined_scores = (1 - pattern_weight) * size_probs + pattern_weight * cluster_scores

        # Incorporate cross-dimensional weights if available
        # Apply quantum price level filtering if enabled
        quantum_weights = None

        if use_quantum_levels and self.quantum_generator is not None:
            # Filter paths based on how well they respect quantum price levels
            quantum_weights = self.quantum_generator.filter_paths_by_levels(
                paths, influence_strength=quantum_influence
            )

        # Apply cross-dimensional filtering if available
        if cross_dim_weights is not None:
            # Calculate average cross-dimensional weight for each cluster
            cluster_cross_weights = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    # Average cross-dimensional weight for this cluster
                    cluster_cross_weights[i] = np.mean(cross_dim_weights[cluster_indices])
                else:
                    cluster_cross_weights[i] = 1.0  # Default if no paths in cluster

            # Normalize to [0, 1] if needed
            if np.sum(cluster_cross_weights) > 0:
                cluster_cross_weights = cluster_cross_weights / np.max(cluster_cross_weights)
            else:
                cluster_cross_weights = np.ones(n_clusters)

            # Blend in cross-dimensional weights (give them 30% influence)
            cross_dim_weight = 0.3
            combined_scores = (1 - cross_dim_weight) * combined_scores + cross_dim_weight * cluster_cross_weights

        # Apply quantum level filtering if available
        if quantum_weights is not None:
            # Calculate average quantum weight for each cluster
            cluster_quantum_weights = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    # Average quantum weight for this cluster
                    cluster_quantum_weights[i] = np.mean(quantum_weights[cluster_indices])
                else:
                    cluster_quantum_weights[i] = 1.0  # Default if no paths in cluster

            # Normalize to [0, 1] if needed
            if np.sum(cluster_quantum_weights) > 0:
                cluster_quantum_weights = cluster_quantum_weights / np.max(cluster_quantum_weights)
            else:
                cluster_quantum_weights = np.ones(n_clusters)

            # Blend in quantum weights (give them 30% influence)
            q_weight = 0.3
            combined_scores = (1 - q_weight) * combined_scores + q_weight * cluster_quantum_weights

        # Normalize probabilities
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
            vol_similarity = abs(path_vol - recent_vol) / max(recent_vol, 1e-8)

            # 2. Return distribution similarity
            path_skew = stats.skew(cloud_returns)
            path_kurt = stats.kurtosis(cloud_returns)
            skew_similarity = abs(path_skew - recent_skew)
            kurt_similarity = abs(path_kurt - recent_kurt)

            # 3. For trading time, add time dilation similarity
            if use_trading_time and enable_time_forecast:
                # Compare path's implied time dilation with recent dilation
                dilation_similarity = 0.0
                # We would calculate this from the path's characteristics, but as an approximation:
                volatility_ratio = path_vol / recent_vol
                # Higher volatility implies faster trading time
                implied_dilation = np.power(volatility_ratio, self.time_warper.alpha)
                dilation_similarity = abs(implied_dilation - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)

                # Combine similarities with trading time component
                total_similarity = (
                    0.5 * vol_similarity +
                    0.2 * skew_similarity +
                    0.1 * kurt_similarity +
                    0.2 * dilation_similarity
                )
            else:
                # Original similarity calculation
                total_similarity = (0.7 * vol_similarity + 0.3 * skew_similarity)

            # Calculate probability - less aggressive decay
            path_probabilities[i] = np.exp(-2 * total_similarity)

        # Distance-based probabilities (alternative approach)
        for i, path in enumerate(cloud_paths):
            distance = np.mean(np.abs(path - most_likely_path))
            path_probabilities[i] = np.exp(-distance / np.std(most_likely_path))

        # Normalize probabilities
        path_probabilities = path_probabilities / np.sum(path_probabilities)

        # Before returning paths, ensure volatility matches historical data
        if preserve_volatility:
            # Calculate historical volatility (day-to-day changes)
            hist_diffs = np.diff(np.log(self.prices))
            hist_std = np.std(hist_diffs)

            # Calculate forecast volatility
            forecast_diffs = np.diff(np.log(paths), axis=1)
            forecast_std = np.mean([np.std(path_diffs) for path_diffs in forecast_diffs])

            # If forecast is too smooth, add appropriate noise
            if forecast_std < 0.8 * hist_std:  # Allow some smoothing, but not too much
                print(f"Adjusting volatility from {forecast_std:.5f} to {hist_std:.5f}")
                volatility_factor = hist_std / forecast_std

                # Add scaled noise to maintain proper volatility
                for i in range(paths.shape[0]):
                    for j in range(1, paths.shape[1]):
                        # Generate noise with same distribution as historical data
                        noise = np.random.choice(hist_diffs) * 0.5  # Scale down slightly for stability
                        # Apply noise multiplicatively
                        paths[i, j] *= np.exp(noise)

        return paths, {
            'labels': labels,
            'cluster_weights': cluster_scores,
            'cluster_sizes': cluster_sizes,
            'pattern_matches': pattern_matches,
            'centroids': centroids,
            'cluster_probs': cluster_probs,
            'most_likely_path': most_likely_path,
            'probability_cloud': cloud_paths,
            'path_probabilities': path_probabilities,
            'trading_time_enabled': use_trading_time,
            'time_dilation_factors': self.time_map['dilation_factors'][-lookback_window:] if use_trading_time else None,
            'cross_dim_enabled': use_cross_dim and cross_dim_weights is not None,
            'cross_dim_regime': cross_dim_regime,
            'cross_dim_weights': cross_dim_weights.tolist() if cross_dim_weights is not None else None,
            'quantum_levels_enabled': use_quantum_levels,
            'quantum_levels': self.quantum_levels,
            'quantum_weights': quantum_weights.tolist() if quantum_weights is not None else None
        }

    def _compute_path_pattern_similarity(self, prices: np.ndarray, patterns: list) -> np.ndarray:
        """Compute similarity between a price path and known patterns."""
        n_patterns = len(patterns)

        # If no patterns, return zero similarities
        if n_patterns == 0:
            return np.zeros(0)

        similarities = np.zeros(n_patterns)

        # For each pattern
        for i, pattern in enumerate(patterns):
            # If pattern is empty or too short, skip it
            if len(pattern) == 0 or len(pattern) < 2:
                similarities[i] = 0
                continue

            # Skip if prices array is too short
            if len(prices) < 2:
                similarities[i] = 0
                continue

            # Normalize pattern to [0, 1] range
            pat_min = np.min(pattern)
            pat_max = np.max(pattern)

            # Avoid division by zero
            if pat_max == pat_min:
                pat_norm = np.zeros_like(pattern)
            else:
                pat_norm = (pattern - pat_min) / (pat_max - pat_min)

            # Ensure pattern is long enough
            min_segment = 10  # Minimum segment length for correlation

            # Use maximum possible segment length
            segment_len = min(min_segment, len(pattern), len(prices))

            # Get segment of prices of the same length as pattern
            segment = prices[-segment_len:]

            # Ensure pat_norm is the right length too
            pat_norm = pat_norm[-segment_len:]

            # Normalize segment to [0, 1] range
            seg_min = np.min(segment)
            seg_max = np.max(segment)

            # Avoid division by zero
            if seg_max == seg_min:
                seg_norm = np.zeros_like(segment)
            else:
                seg_norm = (segment - seg_min) / (seg_max - seg_min)

            # Compute correlation between normalized segment and pattern
            # Add error handling for the correlation coefficient calculation
            try:
                if len(seg_norm) > 1 and len(pat_norm) > 1:
                    corr = np.corrcoef(seg_norm, pat_norm)[0,1]
                    if np.isnan(corr):
                        corr = 0  # Handle NaN correlations
                else:
                    corr = 0  # Not enough data for correlation

                # Scale to similarity: 1 is perfect match, 0 is no match
                similarities[i] = max(0, corr)  # Only positive correlations count as similarity
            except Exception as e:
                print(f"Error in correlation calculation: {e}")
                print(f"Segment shape: {seg_norm.shape}, Pattern shape: {pat_norm.shape}")
                similarities[i] = 0

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

    def simulate_paths_fast(self, n_steps, n_paths=100):
        """Faster path simulation for backtesting with fewer paths."""
        # Simplified version with fewer paths and calculations
        historical_returns = np.diff(np.log(self.prices))
        recent_returns = historical_returns[-min(len(historical_returns), 30):]

        # Use simple sampling with bootstrapping instead of complex regime matching
        sampled_indices = np.random.choice(
            len(recent_returns),
            size=(n_paths, n_steps),
            replace=True
        )

        # Generate paths based on sampled returns
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.prices[-1]  # Start with last price

        for i in range(n_steps):
            # Use the sampled returns for each path
            step_returns = recent_returns[sampled_indices[:, i]]
            paths[:, i+1] = paths[:, i] * np.exp(step_returns)

        # Calculate key statistics
        mean_path = np.mean(paths, axis=0)
        median_path = np.median(paths, axis=0)
        upper_95 = np.percentile(paths, 95, axis=0)
        lower_5 = np.percentile(paths, 5, axis=0)

        # Find most likely path (closest to mean)
        path_diffs = np.sum((paths - mean_path) ** 2, axis=1)
        most_likely_idx = np.argmin(path_diffs)
        most_likely_path = paths[most_likely_idx]

        # Simple analysis with mean path and percentiles
        path_analysis = {
            'mean_path': mean_path,
            'median_path': median_path,
            'most_likely_path': most_likely_path,
            'upper_95': upper_95,
            'lower_5': lower_5
        }

        return paths, path_analysis

    def simulate_paths_gpu(self, n_steps, n_paths=1000):
        """GPU-accelerated path simulation."""
        from fractime.optimization import try_import_cupy

        # Try to import cupy for GPU acceleration
        cp = try_import_cupy()

        if cp is None:
            print("GPU acceleration not available, falling back to CPU")
            return self.simulate_paths_fast(n_steps, n_paths)

        try:
            # Calculate returns
            historical_returns = np.diff(np.log(self.prices))
            recent_returns = historical_returns[-min(len(historical_returns), 30):]

            # Move data to GPU
            recent_returns_gpu = cp.array(recent_returns)

            # Generate paths on GPU
            paths_gpu = cp.zeros((n_paths, n_steps + 1))
            paths_gpu[:, 0] = self.prices[-1]

            # Generate random indices on GPU for bootstrapping
            indices = cp.random.randint(0, len(recent_returns), (n_paths, n_steps))

            # Use GPU for path generation
            for i in range(n_steps):
                returns = recent_returns_gpu[indices[:, i]]
                paths_gpu[:, i+1] = paths_gpu[:, i] * cp.exp(returns)

            # Move results back to CPU
            paths = cp.asnumpy(paths_gpu)

            # Compute statistics on CPU using NumPy
            mean_path = np.mean(paths, axis=0)
            median_path = np.median(paths, axis=0)
            upper_95 = np.percentile(paths, 95, axis=0)
            lower_5 = np.percentile(paths, 5, axis=0)

            # Find most likely path (closest to mean)
            path_diffs = np.sum((paths - mean_path) ** 2, axis=1)
            most_likely_idx = np.argmin(path_diffs)
            most_likely_path = paths[most_likely_idx]

            # Analysis results
            path_analysis = {
                'mean_path': mean_path,
                'median_path': median_path,
                'most_likely_path': most_likely_path,
                'upper_95': upper_95,
                'lower_5': lower_5
            }

            return paths, path_analysis

        except Exception as e:
            print(f"GPU simulation failed: {e}, falling back to CPU")
            return self.simulate_paths_fast(n_steps, n_paths)
