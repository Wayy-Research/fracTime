"""
Quantum finance methods for fractal time series analysis.
Provides quantum-inspired algorithms for financial time series.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import integrate, optimize
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MultidimensionalFractalAnalyzer:
    """Analyzes fractal properties across multiple market dimensions simultaneously."""
    
    def __init__(self, dimensions: int = 2, min_scale: int = 5, max_scale: int = None):
        """
        Initialize multidimensional fractal analyzer.
        
        Args:
            dimensions: Number of market dimensions to analyze
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
        """
        self.dimensions = dimensions
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hurst_matrix = None
        self.cross_correlations = None
        self.attractor = None
    
    def analyze(self, multi_series: List[np.ndarray]) -> Dict:
        """
        Analyze fractal properties across multiple dimensions.
        
        Args:
            multi_series: List of time series representing different market dimensions
            
        Returns:
            Dictionary of multidimensional fractal properties
        """
        if not multi_series or len(multi_series) < 1:
            raise ValueError("At least one time series must be provided")
            
        dimensions = len(multi_series)
        self.dimensions = dimensions
        
        # Ensure all series have the same length
        series_len = len(multi_series[0])
        for i, series in enumerate(multi_series):
            if len(series) != series_len:
                raise ValueError(f"Series at index {i} has different length from series 0")
        
        # Compute Hurst exponent for each series
        hurst_exponents = [self._compute_hurst(series) for series in multi_series]
        
        # Compute cross-correlation matrix
        cross_corr_matrix = np.zeros((dimensions, dimensions))
        for i in range(dimensions):
            for j in range(dimensions):
                if i == j:
                    cross_corr_matrix[i, j] = 1.0
                else:
                    # Compute correlation of returns
                    series_i_returns = np.diff(np.log(multi_series[i]))
                    series_j_returns = np.diff(np.log(multi_series[j]))
                    
                    # Handle possible zeros
                    series_i_returns = np.nan_to_num(series_i_returns)
                    series_j_returns = np.nan_to_num(series_j_returns)
                    
                    if len(series_i_returns) > 1 and len(series_j_returns) > 1:
                        cross_corr_matrix[i, j] = np.corrcoef(series_i_returns, series_j_returns)[0, 1]
                    else:
                        cross_corr_matrix[i, j] = 0.0
        
        # Compute fractal coherence - measure of how aligned fractal patterns are
        fractal_coherence = {}
        
        # Pairwise coherence
        pairwise_coherence = np.zeros((dimensions, dimensions))
        for i in range(dimensions):
            for j in range(dimensions):
                if i == j:
                    pairwise_coherence[i, j] = 1.0
                else:
                    # Compute similarity in fractal scaling patterns
                    # Simple version: similarity in Hurst exponents modified by correlation
                    h_diff = 1.0 - abs(hurst_exponents[i] - hurst_exponents[j])
                    corr = abs(cross_corr_matrix[i, j])
                    pairwise_coherence[i, j] = h_diff * np.sqrt(corr)
        
        # Overall coherence (average of all pairwise)
        coherence_values = []
        for i in range(dimensions):
            for j in range(i+1, dimensions):
                coherence_values.append(pairwise_coherence[i, j])
                
        overall_coherence = np.mean(coherence_values) if coherence_values else 0.0
        
        # Store the results
        self.hurst_matrix = hurst_exponents
        self.cross_correlations = cross_corr_matrix
        
        # Extract attractor properties if enough dimensions
        attractor = None
        if dimensions >= 2:
            attractor = self._extract_attractor(multi_series)
            self.attractor = attractor
        
        # Determine market regime based on hurst values
        regime_type = 0  # Default: trending (0=trending, 1=mean-reverting, 2=random)
        primary_h = hurst_exponents[0]  # Assume first series is price
        
        if primary_h < 0.45:
            regime_type = 1  # Mean-reverting
        elif 0.45 <= primary_h <= 0.55:
            regime_type = 2  # Random walk / efficient
        else:
            regime_type = 0  # Trending / persistent
            
        # Prepare and return results
        return {
            'hurst_exponents': hurst_exponents,
            'cross_correlation': cross_corr_matrix.tolist(),
            'fractal_coherence': {
                'pairwise': pairwise_coherence.tolist(),
                'overall': float(overall_coherence)
            },
            'regime': {
                'regime': regime_type,
                'primary_hurst': float(primary_h)
            },
            'attractor': attractor
        }

    def _compute_hurst(self, series: np.ndarray) -> float:
        """
        Compute Hurst exponent for a time series.
        
        Args:
            series: Time series data
            
        Returns:
            Hurst exponent
        """
        # Make sure series is not zero
        if np.all(series == 0):
            return 0.5  # Return random walk value if all zeros
            
        # Convert to log returns for better behavior
        try:
            # Use np.log1p(series) if needed to avoid log(0) issues
            log_returns = np.diff(np.log(np.maximum(series, 1e-10)))
        except Exception:
            # Fallback to raw series if log returns fail
            log_returns = np.diff(series)
        
        # Remove NaNs or infinities
        log_returns = np.nan_to_num(log_returns)
        
        # Need at least 8 points for valid calculation
        if len(log_returns) < 8:
            return 0.5  # Return random walk value if too short
        
        # Rescaled range method for Hurst exponent
        tau = []
        lagvec = []
        
        # Use shorter subseries for long series
        max_lag = min(100, len(log_returns) // 4)
        
        # Use powers of 2 for lags
        lags = np.unique(np.logspace(0.7, np.log10(max_lag), 20).astype(int))
        lags = lags[lags >= 4]  # Ensure minimum lag of 4
        
        for lag in lags:
            # Skip if too few points
            if lag >= len(log_returns):
                continue
                
            # Compute R/S for this lag
            rs_values = []
            
            # Break series into subseries of length lag
            n_subsets = len(log_returns) // lag
            
            if n_subsets < 1:
                continue
                
            for i in range(n_subsets):
                subset = log_returns[i*lag:(i+1)*lag]
                
                # Compute R/S statistic for this subset
                mean_subset = np.mean(subset)
                std_subset = np.std(subset)
                
                # Avoid division by zero
                if std_subset == 0:
                    continue
                    
                # Compute cumulative deviation from mean
                cumdev = np.cumsum(subset - mean_subset)
                
                # R = max(cumdev) - min(cumdev)
                r_value = np.max(cumdev) - np.min(cumdev)
                
                # S = standard deviation
                s_value = std_subset
                
                # R/S value
                rs = r_value / s_value if s_value > 0 else 0
                
                if rs > 0:
                    rs_values.append(rs)
            
            # If we have R/S values for this lag
            if rs_values:
                tau.append(np.mean(rs_values))
                lagvec.append(lag)
        
        # Compute Hurst exponent as slope of log-log plot
        if len(tau) > 1 and len(lagvec) > 1:
            # Log-log regression
            log_lagvec = np.log10(lagvec)
            log_tau = np.log10(tau)
            
            # Compute slope through linear regression
            try:
                hurst = np.polyfit(log_lagvec, log_tau, 1)[0]
                
                # Clamp to reasonable range [0.1, 0.9]
                hurst = min(0.9, max(0.1, hurst))
                
                return hurst
            except Exception:
                # Return 0.5 if regression fails
                return 0.5
        else:
            # Not enough points
            return 0.5

    def _extract_attractor(self, series: List[np.ndarray]) -> Dict:
        """
        Extract attractor properties from multi-dimensional data.
        
        Args:
            series: List of time series
            
        Returns:
            Dictionary of attractor properties
        """
        if len(series) < 2:
            return {"error": "Need at least 2 dimensions for attractor analysis"}
        
        # Extract main series (first 2 dimensions)
        dim1 = series[0]
        dim2 = series[1]
        
        # Convert to returns if needed
        dim1_returns = np.diff(np.log(np.maximum(dim1, 1e-10)))
        dim2_returns = np.diff(np.log(np.maximum(dim2, 1e-10)))
        
        # Clean data
        dim1_returns = np.nan_to_num(dim1_returns)
        dim2_returns = np.nan_to_num(dim2_returns)
        
        # Use the returns for phase space reconstruction
        points = np.column_stack([dim1_returns, dim2_returns])
        
        # Add 3rd dimension if 3 or more series provided
        if len(series) >= 3:
            dim3 = series[2]
            dim3_returns = np.diff(np.log(np.maximum(dim3, 1e-10)))
            dim3_returns = np.nan_to_num(dim3_returns)
            points = np.column_stack([points, dim3_returns])
        
        # Compute attractor properties
        
        # 1. Centroid
        centroid = np.mean(points, axis=0)
        
        # 2. Spread/Extent - use standard deviation
        spread = np.std(points, axis=0)
        
        # 3. Density estimation - use histogram
        hist_range = 2.0  # +/- 2 standard deviations
        nbins = 10
        
        # Create histogram for first two dimensions
        hist, _ = np.histogramdd(
            points[:, :2],
            bins=nbins,
            range=[
                [centroid[0] - hist_range * spread[0], centroid[0] + hist_range * spread[0]],
                [centroid[1] - hist_range * spread[1], centroid[1] + hist_range * spread[1]]
            ]
        )
        
        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        # 4. Hotspots - find highest density regions
        flat_hist = hist.flatten()
        indices = np.argsort(flat_hist)[::-1]  # Sort in descending order
        
        # Take top 20% as hotspots
        n_hotspots = max(1, int(0.2 * nbins * nbins))
        hotspot_indices = indices[:n_hotspots]
        
        # Convert flattened indices to 2D
        hotspots = []
        for idx in hotspot_indices:
            i, j = np.unravel_index(idx, hist.shape)
            
            # Skip if very low density
            if flat_hist[idx] < 0.01:
                continue
                
            # Compute center of this bin
            x_center = centroid[0] - hist_range * spread[0] + (i + 0.5) * (2 * hist_range * spread[0] / nbins)
            y_center = centroid[1] - hist_range * spread[1] + (j + 0.5) * (2 * hist_range * spread[1] / nbins)
            
            hotspots.append({
                'center': [float(x_center), float(y_center)],
                'density': float(flat_hist[idx])
            })
        
        # 5. Compute fractal dimension of attractor
        # Use box-counting method on histogram
        box_sizes = []
        box_counts = []
        
        # Try different box sizes
        for box_size in range(2, min(5, nbins)):
            if nbins % box_size != 0:
                continue
                
            # Count non-empty boxes
            n_boxes = (nbins // box_size) ** 2
            boxes_filled = 0
            
            # Group histogram bins into larger boxes
            for i in range(0, nbins, box_size):
                for j in range(0, nbins, box_size):
                    if np.sum(hist[i:i+box_size, j:j+box_size]) > 0:
                        boxes_filled += 1
            
            box_sizes.append(box_size)
            box_counts.append(boxes_filled)
        
        # Compute fractal dimension from log-log plot
        attractor_dim = 2.0  # Default if we can't compute
        if len(box_sizes) > 1:
            try:
                log_box_sizes = np.log(1.0 / np.array(box_sizes))
                log_box_counts = np.log(np.array(box_counts))
                slope = np.polyfit(log_box_sizes, log_box_counts, 1)[0]
                attractor_dim = slope
                
                # Clamp to reasonable range
                attractor_dim = min(3.0, max(1.0, attractor_dim))
            except Exception:
                pass
        
        return {
            'centroid': centroid.tolist(),
            'spread': spread.tolist(),
            'fractal_dimension': float(attractor_dim),
            'hotspots': hotspots,
            'histogram': hist.tolist()
        }

class QuantumPriceLevels:
    """Generates quantum price levels based on quantum mechanical principles."""

    def __init__(self, num_levels=7, period=30):
        """
        Initialize the quantum price level calculator.
        
        Args:
            num_levels: Number of price levels to generate
            period: Time period for the price level calculation
        """
        self.num_levels = num_levels
        self.period = period
        self._generator = QuantumPriceLevelGenerator(time_horizon=period/252, energy_levels=num_levels)
        
    def calculate_levels(self, prices):
        """
        Calculate quantum price levels from price data.
        
        Args:
            prices: Array of price data
        
        Returns:
            Array of calculated price levels
        """
        result = self._generator.generate_price_levels(prices)
        return [level["price"] for level in result["levels"]]


class QuantumPriceLevelGenerator:
    """
    Generates quantum price levels based on quantum mechanical principles.
    
    The Quantum Price Level approach applies concepts from quantum physics
    to financial markets, modeling price barriers as energy states in a
    quantum mechanical system.
    
    Key concepts:
    - Price distribution is transformed into a quantum potential function
    - Price levels emerge naturally as quantum energy states (eigenvalues)
    - Wave functions (eigenvectors) show where price is likely to stabilize
    - Quantum tunneling explains how price can break through barriers
    - Uncertainty principle reflects price level width/significance tradeoff
    """
    
    def __init__(self, time_horizon: float = 0.5, energy_levels: int = 3, alpha: float = 0.5):
        """
        Initialize quantum price level generator.
        
        Args:
            time_horizon: Time horizon for price level projection (years)
            energy_levels: Number of quantum energy levels to consider
            alpha: Power-law exponent for quantum potential function
        """
        self.time_horizon = time_horizon
        self.energy_levels = energy_levels  # Default reduced from 5 to 3 for better convergence
        self.alpha = alpha
        self.price_levels = None
        self.potential_function = None
        self.wave_function = None
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate annualized volatility from price history."""
        log_returns = np.diff(np.log(prices))
        return np.std(log_returns) * np.sqrt(252)  # Annualize
    
    def _calculate_potential_function(
        self,
        prices: np.ndarray,
        num_points: int = 200,  # Reduced from 1000 to improve convergence
        smoothing: float = 0.3  # Increased from 0.2 for smoother potential
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct quantum potential function from price density.
        
        The potential function is created from the price density:
        - Areas where prices spend more time (high density) become low potential wells
        - Areas rarely visited by price become high potential barriers
        - This creates a quantum landscape where price tends to be "trapped" in wells
        
        Args:
            prices: Historical price data
            num_points: Number of points for potential discretization
            smoothing: Smoothing parameter for density estimation
            
        Returns:
            Tuple of (price_grid, potential_values)
        """
        # Define price range with some margin
        price_range = np.max(prices) - np.min(prices)
        p_min = np.min(prices) - 0.1 * price_range
        p_max = np.max(prices) + 0.1 * price_range
        
        # Create price grid - use fewer points for better solver convergence
        price_grid = np.linspace(p_min, p_max, num_points)
        
        # Calculate kernel density estimate (KDE) of price distribution
        from scipy import stats
        
        # Add some noise to prices to prevent singular matrices
        jittered_prices = prices + np.random.normal(0, price_range * 0.0001, size=prices.shape)
        
        # Create KDE with more smoothing for better convergence
        try:
            kde = stats.gaussian_kde(jittered_prices, bw_method=smoothing)
            density = kde(price_grid)
        except Exception as e:
            print(f"KDE estimation failed: {e}, using simple histogram instead")
            # Fallback to histogram for density estimation
            hist, bin_edges = np.histogram(prices, bins=num_points//2, range=(p_min, p_max), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Interpolate to get density at our grid points
            density = np.interp(price_grid, bin_centers, hist)
        
        # Smooth density further if needed
        if np.max(density) - np.min(density) > 1e3:
            print("Extreme density values detected, applying additional smoothing")
            # Apply moving average smoothing
            window_size = max(3, num_points // 20)  # 5% of points for window
            kernel = np.ones(window_size) / window_size
            density = np.convolve(density, kernel, mode='same')
        
        # Normalize density to [0, 1]
        if np.max(density) > 0:
            density = density / np.max(density)
        else:
            density = np.ones_like(density) * 0.5  # Uniform if all zeros
        
        # Transform density to potential energy
        # Areas of high price density become low potential energy wells
        # Using inverse power relationship with alpha parameter
        # Apply a more moderate transformation for better numerics
        potential = (1 - density) ** min(self.alpha, 0.7)  # Cap alpha for numerical stability
        
        # Ensure potential is well-behaved
        potential = np.clip(potential, 0.01, 0.99)  # Avoid extreme values
        
        # Store the potential function
        self.potential_function = (price_grid, potential)
        
        return price_grid, potential
    
    def _solve_schrodinger_equation(
        self,
        price_grid: np.ndarray,
        potential: np.ndarray,
        num_eigenvalues: int = 5  # Reduced default
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve time-independent Schrödinger equation to find quantum price levels.
        
        This applies quantum mechanics principles to financial markets:
        - The price grid represents possible price points
        - The potential function represents the "quantum potential energy landscape"
        - Areas with low potential (high price density) correspond to stable price zones
        - The eigenvalues and eigenvectors represent the quantum states and their probabilities
        - These quantum states manifest as support/resistance price levels
        
        Args:
            price_grid: Grid of price points
            potential: Quantum potential at each price point
            num_eigenvalues: Number of eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Immediately limit the number of eigenvalues to improve convergence chances
        num_eigenvalues = min(5, num_eigenvalues)  # Hard limit at 5
        
        # Parameters for the quantum system
        hbar = 1.0  # Normalized Planck constant
        mass = 1.0  # Normalized effective mass
        
        # Calculate grid spacing
        dx = price_grid[1] - price_grid[0]
        
        # Build Hamiltonian matrix (sparse for efficiency)
        n = len(price_grid)
        
        # Skip quantum calculation if grid is too large - use fallback instead
        if n > 2000:
            print("Price grid too large for quantum solver, using fallback method directly")
            return self._create_fallback_eigen(price_grid, potential, num_eigenvalues)
        
        # Normalize potential to improve numerical stability
        potential_min = np.min(potential)
        potential_max = np.max(potential)
        potential_range = potential_max - potential_min
        
        if potential_range > 0:
            normalized_potential = (potential - potential_min) / potential_range
        else:
            normalized_potential = np.zeros_like(potential)
            
        # For very flat potentials, add some random noise to help convergence
        if potential_range < 0.01:
            print("Adding noise to flat potential to aid convergence")
            normalized_potential += np.random.normal(0, 0.001, size=normalized_potential.shape)
        
        # Laplacian part (kinetic energy) using finite difference
        diag = np.ones(n) * 2.0
        offdiag = np.ones(n-1) * (-1.0)
        laplacian = sparse.diags([offdiag, diag, offdiag], [-1, 0, 1])
        
        # Scale by constants - reduce coefficient for better conditioning
        kinetic = -0.1 * hbar**2 / mass * laplacian / dx**2  # Reduced coefficient
        
        # Potential energy part
        potential_diag = sparse.diags([normalized_potential], [0])
        
        # Total Hamiltonian
        hamiltonian = kinetic + potential_diag
        
        # Regularize the Hamiltonian to help convergence
        identity = sparse.eye(n)
        regularized_hamiltonian = hamiltonian + 0.001 * identity
        
        # Number of eigenvalues to calculate - cannot exceed matrix dimension
        num_eigenvalues = min(num_eigenvalues, n-2)  
        
        # Skip ARPACK for smaller matrices - go straight to dense solver
        if n <= 200:
            try:
                print("Using dense solver for small matrix")
                dense_hamiltonian = regularized_hamiltonian.toarray()
                eigenvalues_full, eigenvectors_full = np.linalg.eigh(dense_hamiltonian)
                
                # Take the smallest num_eigenvalues
                eigenvalues = eigenvalues_full[:num_eigenvalues]
                eigenvectors = eigenvectors_full[:, :num_eigenvalues]
                return eigenvalues, eigenvectors
            except Exception as e:
                print(f"Dense solver failed: {e}")
                return self._create_fallback_eigen(price_grid, potential, num_eigenvalues)
        
        # For larger matrices, try ARPACK with improved settings
        try:
            # First attempt with very relaxed settings
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                regularized_hamiltonian, k=num_eigenvalues, which='SM', 
                maxiter=1000,  # Reduced iterations
                tol=1e-3,      # Relaxed tolerance
                ncv=max(20, 3*num_eigenvalues),  # More Lanczos vectors
                sigma=0.1      # Shift to find lowest eigenvalues
            )
        except Exception as e:
            print(f"Eigenvalue solver failed: {e}")
            
            # Go straight to fallback method
            return self._create_fallback_eigen(price_grid, potential, num_eigenvalues)
            
        # Scale eigenvalues back to original range if needed
        if potential_range > 0:
            eigenvalues = eigenvalues * potential_range + potential_min
        
        return eigenvalues, eigenvectors
        
    def _create_fallback_eigen(
        self, 
        price_grid: np.ndarray, 
        potential: np.ndarray, 
        num_eigenvalues: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create fallback eigenvalues and eigenvectors when solver fails.
        Uses potential minima to create approximate quantum states.
        
        Args:
            price_grid: Grid of price points
            potential: Quantum potential at each price point
            num_eigenvalues: Number of eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) as fallback
        """
        print("Using fallback eigenvalue/eigenvector method")
        n = len(price_grid)
        
        # Find local minima in potential (these are natural eigenstates)
        minima_indices = []
        
        # Simple minima finding
        for i in range(1, n-1):
            if potential[i] < potential[i-1] and potential[i] < potential[i+1]:
                minima_indices.append(i)
        
        # If we didn't find enough minima, add some based on lowest potential
        if len(minima_indices) < num_eigenvalues:
            # Find indices of lowest potential values
            sorted_indices = np.argsort(potential)
            # Add until we have enough, skipping duplicates
            for idx in sorted_indices:
                if idx not in minima_indices:
                    minima_indices.append(idx)
                    if len(minima_indices) >= num_eigenvalues:
                        break
        
        # Take only the required number
        if len(minima_indices) > num_eigenvalues:
            # Sort by potential value (lowest first)
            minima_indices = sorted(minima_indices, key=lambda i: potential[i])[:num_eigenvalues]
        
        # Create eigenvalues based on potential at minima
        eigenvalues = np.array([potential[i] for i in minima_indices])
        
        # Create approximate eigenvectors (Gaussian peaks at minima)
        eigenvectors = np.zeros((n, len(minima_indices)))
        
        for j, idx in enumerate(minima_indices):
            # Create Gaussian centered at minimum
            width = max(5, n // 20)  # Width proportional to grid size
            for i in range(n):
                # Gaussian function centered at minimum
                eigenvectors[i, j] = np.exp(-0.5 * ((i - idx) / width)**2)
            
            # Normalize
            eigenvectors[:, j] = eigenvectors[:, j] / np.linalg.norm(eigenvectors[:, j])
        
        return eigenvalues, eigenvectors
    
    def _compute_probabilities(
        self, 
        eigenvectors: np.ndarray, 
        prices: np.ndarray,
        price_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute transition probabilities between energy levels.
        
        Args:
            eigenvectors: Eigenvectors of the Hamiltonian
            prices: Historical price data
            price_grid: Grid of price points
            
        Returns:
            Array of probabilities for each energy level
        """
        # Find index of current price in grid
        current_price = prices[-1]
        idx = np.argmin(np.abs(price_grid - current_price))
        
        # Check if we have valid eigenvectors
        if eigenvectors.shape[1] == 0:
            # No eigenvectors available (fallback)
            return np.array([1.0])  # Return single probability
        
        # Extract probability amplitudes at current price
        n_levels = eigenvectors.shape[1]
        probabilities = np.zeros(n_levels)
        
        # Get probability amplitudes
        try:
            for i in range(n_levels):
                # Square amplitude gives probability
                probabilities[i] = eigenvectors[idx, i]**2
                
            # Normalize probabilities
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            else:
                # If all probabilities are zero, use uniform distribution
                probabilities = np.ones(n_levels) / n_levels
                
        except Exception as e:
            print(f"Error computing probabilities: {e}")
            # Fallback to uniform distribution
            probabilities = np.ones(n_levels) / n_levels
        
        return probabilities
    
    def generate_price_levels(self, price_history: np.ndarray) -> Dict:
        """
        Generate quantum price levels from historical price data.
        
        This method applies quantum mechanics to identify key price levels:
        
        1. First, we create a quantum potential landscape from price density
           - Areas where price spends more time become potential "wells"
           - Areas rarely visited become potential "barriers"
           
        2. Then we solve the Schrödinger equation in this potential landscape
           - This models price as a quantum particle in the potential field
           - The solution gives us energy states (eigenvalues) & wave functions (eigenvectors)
           
        3. Each energy state corresponds to a quantum price level
           - The lowest energy states are the strongest support/resistance levels
           - Wave function peaks show where price is most likely to stabilize
           - Wave function width reflects uncertainty in the price level
        
        4. These quantum price levels provide mathematically rigorous
           support/resistance levels based on market's collective behavior
        
        Args:
            price_history: Historical price data
            
        Returns:
            Dictionary with quantum price levels information
        """
        # Calculate market volatility
        volatility = self._calculate_volatility(price_history)
        
        # Generate quantum potential
        price_grid, potential = self._calculate_potential_function(price_history)
        
        # Solve Schrödinger equation
        eigenvalues, eigenvectors = self._solve_schrodinger_equation(
            price_grid, potential, num_eigenvalues=self.energy_levels
        )
        
        # Store wave function
        self.wave_function = eigenvectors
        
        # Compute transition probabilities
        probabilities = self._compute_probabilities(eigenvectors, price_history, price_grid)
        
        # Extract quantum price levels
        levels = []
        
        # Use only the available number of energy levels (might be fewer than self.energy_levels)
        actual_energy_levels = min(self.energy_levels, eigenvectors.shape[1])
        
        for i in range(actual_energy_levels):
            # Find maximum amplitude of wave function
            psi = eigenvectors[:, i]
            max_idx = np.argmax(np.abs(psi))
            price = price_grid[max_idx]
            
            # Compute level width (uncertainty)
            # Find points where wave function drops to half max
            max_psi = np.abs(psi[max_idx])
            half_max_indices = np.where(np.abs(psi) >= 0.5 * max_psi)[0]
            
            if len(half_max_indices) >= 2:
                width = price_grid[half_max_indices[-1]] - price_grid[half_max_indices[0]]
            else:
                width = volatility * price  # Default width based on volatility
            
            # Store price level information
            levels.append({
                'level': i,
                'price': float(price),
                'energy': float(eigenvalues[i]),
                'probability': float(probabilities[i] if i < len(probabilities) else 1.0/actual_energy_levels),
                'width': float(width)
            })
        
        # If we have fewer levels than expected, fill in with simple levels based on price
        if actual_energy_levels < self.energy_levels:
            print(f"Warning: Only {actual_energy_levels} energy levels available, adding {self.energy_levels - actual_energy_levels} simple levels")
            
            # Get price range
            min_price = np.min(price_history) * 0.95
            max_price = np.max(price_history) * 1.05
            price_range = max_price - min_price
            
            # Add evenly spaced levels
            existing_prices = [level['price'] for level in levels]
            
            for i in range(actual_energy_levels, self.energy_levels):
                # Try to find price that's not too close to existing ones
                attempts = 0
                new_price = None
                
                while attempts < 10:
                    # Generate price at a key Fibonacci level
                    fib_level = (i - actual_energy_levels + 1) / (self.energy_levels - actual_energy_levels + 1)
                    if fib_level < 0.382:
                        # Support below current price
                        new_price = price_history[-1] - fib_level * price_range * 0.5
                    else:
                        # Resistance above current price  
                        new_price = price_history[-1] + (fib_level - 0.382) * price_range * 0.5
                    
                    # Check if this price is far enough from existing ones
                    too_close = False
                    for p in existing_prices:
                        if abs(new_price - p) < 0.02 * price_range:
                            too_close = True
                            break
                            
                    if not too_close:
                        break
                        
                    attempts += 1
                
                # If we couldn't find a good price, just use a random one
                if new_price is None or attempts >= 10:
                    new_price = min_price + np.random.random() * price_range
                
                # Add this price level
                width = volatility * new_price
                levels.append({
                    'level': i,
                    'price': float(new_price),
                    'energy': float(i + 1),  # Higher energy for supplementary levels
                    'probability': float(0.5 / self.energy_levels),  # Lower probability
                    'width': float(width)
                })
                
                existing_prices.append(new_price)
        
        # Sort levels by price
        levels = sorted(levels, key=lambda x: x['price'])
        
        # Calculate support and resistance strength
        for level in levels:
            # Higher probability = stronger level
            level['strength'] = level['probability'] * (1 - min(1.0, level['width'] / (0.1 * price_history[-1])))
        
        # Store the results
        self.price_levels = {
            'levels': levels,
            'current_price': float(price_history[-1]),
            'volatility': float(volatility),
            'time_horizon': self.time_horizon
        }
        
        return self.price_levels
    
    def filter_paths_by_levels(
        self, 
        paths: np.ndarray, 
        influence_strength: float = 1.0
    ) -> np.ndarray:
        """
        Filter paths based on how well they respect quantum price levels.
        
        Args:
            paths: Array of price paths (n_paths, n_steps)
            influence_strength: How strongly levels influence paths (0-1)
            
        Returns:
            Array of path weights
        """
        if self.price_levels is None:
            print("Warning: price_levels not generated yet, returning uniform weights")
            return np.ones(paths.shape[0])
            
        # Safety check for levels
        if not self.price_levels.get('levels'):
            print("Warning: No price levels found, returning uniform weights")
            return np.ones(paths.shape[0])
            
        levels = self.price_levels['levels']
        n_paths, n_steps = paths.shape
        
        # Initialize path weights (higher = more respect for quantum levels)
        path_weights = np.ones(n_paths)
        
        try:
            # Calculate average inter-level distance
            level_prices = [level['price'] for level in levels]
            if len(level_prices) >= 2:
                avg_distance = np.mean(np.diff(sorted(level_prices)))
            else:
                # If we only have one level, use volatility-based distance
                avg_distance = self.price_levels.get('volatility', 0.01) * self.price_levels.get('current_price', 100.0)
            
            # Examine each path's interaction with quantum levels
            for p in range(n_paths):
                path = paths[p]
                
                # Track path crossings through quantum levels
                crossings = 0
                bounces = 0
                
                for t in range(1, n_steps):
                    # Check each quantum level
                    for level in levels:
                        price = level.get('price', 0)
                        width = level.get('width', 1.0)
                        strength = level.get('strength', 0.5)
                        
                        # Check if path crossed this level
                        if (path[t-1] < price and path[t] > price) or (path[t-1] > price and path[t] < price):
                            crossings += 1
                        
                        # Check if path bounced off this level
                        # Path approaches level then reverses direction
                        if abs(path[t-1] - price) < width and np.sign(path[t] - path[t-1]) != np.sign(path[t-1] - price):
                            bounces += 1 * strength  # Weight by level strength
                
                # Compute quantum level respect score
                # Paths with appropriate bounces and fewer random crossings score higher
                
                # Expected crossings for a random path
                volatility = self.price_levels.get('volatility', 0.01)  # Default if missing
                expected_crossings = n_steps * len(levels) / max(avg_distance / volatility, 0.1)
                expected_crossings = max(1, expected_crossings)  # Ensure non-zero
                
                # Crossing penalty (penalize excessive random crossings)
                crossing_score = max(0, 1 - (crossings / expected_crossings))
                
                # Bounce bonus (reward respecting quantum levels as support/resistance)
                bounce_bonus = min(1.0, bounces / max(1, len(levels) * 0.5))
                
                # Combine scores, with bounces weighted more heavily
                level_respect_score = 0.3 * crossing_score + 0.7 * bounce_bonus
                
                # Update path weight
                path_weights[p] = 1.0 + influence_strength * (level_respect_score - 0.5)
                
                # Ensure weights stay positive
                path_weights[p] = max(0.1, path_weights[p])
            
        except Exception as e:
            print(f"Error in filter_paths_by_levels: {e}")
            # Return uniform weights in case of error
            return np.ones(n_paths)
            
        # Normalize weights
        if np.sum(path_weights) > 0:
            path_weights = path_weights / np.max(path_weights)
        
        return path_weights

    def visualize_price_levels(
        self, 
        price_history: np.ndarray, 
        future_paths: np.ndarray = None, 
        dates: np.ndarray = None,
        path_weights: np.ndarray = None,
        top_path: np.ndarray = None
    ) -> go.Figure:
        """
        Visualize quantum price levels and potential paths.
        
        Args:
            price_history: Historical price data
            future_paths: Optional forecasted price paths
            dates: Optional dates array
            path_weights: Optional weights for future paths
            top_path: Optional most likely path to highlight
            
        Returns:
            Plotly figure with visualization
        """
        if self.price_levels is None:
            raise ValueError("Must call generate_price_levels first")
            
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Price with Quantum Levels",
                "Quantum Potential Function",
                "Price Level Wave Functions",
                "Price Level Strength"
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "bar"}],
            ],
            row_heights=[0.6, 0.4]
        )
        
        # Create x-axis (dates or indices)
        if dates is None:
            dates = np.arange(len(price_history))
            
        # If we have future paths, extend dates
        if future_paths is not None:
            n_future = future_paths.shape[1]
            last_date = dates[-1]
            
            # Check if dates are datetime
            if isinstance(last_date, np.datetime64) or hasattr(last_date, 'day'):
                # Create business day sequence
                import pandas as pd
                future_dates = pd.date_range(
                    start=last_date, 
                    periods=n_future+1, 
                    freq='B'
                )[1:]  # Skip first date (last historical date)
                extended_dates = np.concatenate([dates, future_dates])
            else:
                # Numeric dates, just extend the sequence
                future_dates = np.arange(last_date + 1, last_date + n_future + 1)
                extended_dates = np.concatenate([dates, future_dates])
        else:
            extended_dates = dates
            
        # 1. Plot price with quantum levels
        # Add historical price
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=price_history,
                name="Historical Price",
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add future paths if available
        if future_paths is not None:
            n_paths = min(30, future_paths.shape[0])  # Limit number of displayed paths
            
            # Sample paths to display
            indices = np.linspace(0, future_paths.shape[0]-1, n_paths, dtype=int)
            
            for i in indices:
                path = future_paths[i]
                
                # Determine color based on weight if available
                if path_weights is not None:
                    weight = path_weights[i]
                    # Scale weight to determine color (green = high weight, gray = low weight)
                    if weight > 0.7:
                        color = f'rgba(0,128,0,{min(1.0, weight)})'  # Green
                    elif weight > 0.4:
                        color = f'rgba(180,180,0,{min(1.0, weight)})'  # Yellow
                    else:
                        color = f'rgba(200,200,200,{max(0.3, weight)})'  # Gray
                else:
                    color = 'rgba(200,200,200,0.3)'  # Default gray
                
                # Create extended path starting from last historical price
                full_path = np.concatenate([[price_history[-1]], path])
                path_dates = extended_dates[-len(full_path):]
                
                fig.add_trace(
                    go.Scatter(
                        x=path_dates,
                        y=full_path,
                        name=f"Path {i}",
                        line=dict(color=color, width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add top path if provided
            if top_path is not None:
                full_top_path = np.concatenate([[price_history[-1]], top_path])
                top_path_dates = extended_dates[-len(full_top_path):]
                
                fig.add_trace(
                    go.Scatter(
                        x=top_path_dates,
                        y=full_top_path,
                        name="Most Likely Path",
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )
        
        # Add quantum price levels as horizontal lines
        for level in self.price_levels['levels']:
            price = level['price']
            strength = level['strength']
            width = level['width']
            
            # Color based on probability (strength)
            color = f'rgba(128,0,128,{min(0.8, 0.3 + strength * 0.7)})'  # Purple with varying opacity
            
            # Add horizontal line for the level
            fig.add_trace(
                go.Scatter(
                    x=[dates[0], extended_dates[-1]],
                    y=[price, price],
                    name=f"QPL: {price:.2f}",
                    line=dict(
                        color=color,
                        width=2 + 3 * strength,  # Width based on strength
                        dash='dash'
                    )
                ),
                row=1, col=1
            )
            
            # Add band for level width
            if width > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[dates[0], extended_dates[-1], extended_dates[-1], dates[0], dates[0]],
                        y=[price + width/2, price + width/2, price - width/2, price - width/2, price + width/2],
                        fill="toself",
                        fillcolor=f'rgba(128,0,128,{min(0.3, 0.1 + strength * 0.2)})',
                        line=dict(width=0),
                        showlegend=False,
                        name=f"QPL Band: {price:.2f}"
                    ),
                    row=1, col=1
                )
        
        # 2. Quantum Potential Function
        if self.potential_function is not None:
            price_grid, potential = self.potential_function
            
            fig.add_trace(
                go.Scatter(
                    x=price_grid,
                    y=potential,
                    name="Quantum Potential",
                    line=dict(color='purple', width=2)
                ),
                row=1, col=2
            )
            
            # Add current price marker
            current_price = price_history[-1]
            idx = np.argmin(np.abs(price_grid - current_price))
            current_potential = potential[idx]
            
            fig.add_trace(
                go.Scatter(
                    x=[current_price],
                    y=[current_potential],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='circle'
                    ),
                    name="Current Price"
                ),
                row=1, col=2
            )
            
            # Add quantum price level markers on potential
            for level in self.price_levels['levels']:
                price = level['price']
                idx = np.argmin(np.abs(price_grid - price))
                level_potential = potential[idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=[price],
                        y=[level_potential],
                        mode='markers',
                        marker=dict(
                            color='green',
                            size=8,
                            symbol='star'
                        ),
                        name=f"Level: {price:.2f}"
                    ),
                    row=1, col=2
                )
        
        # 3. Wave Functions
        if self.wave_function is not None and self.potential_function is not None:
            price_grid, _ = self.potential_function
            eigenvectors = self.wave_function
            
            # Plot first few wave functions
            n_waves = min(3, eigenvectors.shape[1])
            
            for i in range(n_waves):
                # Normalize for better visualization
                psi = eigenvectors[:, i]
                psi = psi / np.max(np.abs(psi)) * 0.5
                
                fig.add_trace(
                    go.Scatter(
                        x=price_grid,
                        y=psi,
                        name=f"Wave Function {i}",
                        line=dict(
                            color=['blue', 'green', 'orange'][i],
                            width=2
                        )
                    ),
                    row=2, col=1
                )
        
        # 4. Level Strength Bar Chart
        levels = self.price_levels['levels']
        
        fig.add_trace(
            go.Bar(
                x=[f"QPL {i+1}: {level['price']:.2f}" for i, level in enumerate(levels)],
                y=[level['strength'] for level in levels],
                name="Level Strength",
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Quantum Price Level Analysis",
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        fig.update_xaxes(title_text="Price", row=1, col=2)
        fig.update_yaxes(title_text="Potential Energy", row=1, col=2)
        
        fig.update_xaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Wave Function", row=2, col=1)
        
        fig.update_xaxes(title_text="Quantum Price Level", row=2, col=2)
        fig.update_yaxes(title_text="Strength", row=2, col=2)
        
        return fig
        
        
def demo_quantum_fractal_analysis(symbol: str = "AAPL", start_date: str = None, n_steps: int = 30) -> dict:
    """
    Run a demonstration of quantum fractal analysis on a financial time series.
    
    Args:
        symbol: Stock symbol to analyze
        start_date: Start date for data retrieval (default: 1 year ago)
        n_steps: Number of forecast steps
        
    Returns:
        Dictionary containing analysis results
    """
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Set default start date if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Download data
    data = yf.download(symbol, start=start_date)
    prices = data['Close'].to_numpy()
    dates = data.index.to_numpy()
    
    # Create quantum analyzer
    qpl_generator = QuantumPriceLevelGenerator()
    
    # Generate quantum price levels
    qpl_results = qpl_generator.generate_price_levels(prices)
    
    # Create multidimensional analyzer
    multi_analyzer = MultidimensionalFractalAnalyzer()
    
    # Prepare data series (price and volume)
    price_series = prices
    volume_series = data['Volume'].to_numpy()
    
    # Normalize volume to be on similar scale as price
    norm_volume = volume_series / np.mean(volume_series) * np.mean(price_series)
    
    # Run multidimensional analysis
    multi_results = multi_analyzer.analyze([price_series, norm_volume])
    
    # Create a basic forecast (example only)
    # For a real forecast, you would use a more sophisticated method
    from scipy import stats
    
    # Compute log returns
    log_returns = np.diff(np.log(prices))
    
    # Generate random paths using bootstrap of historical returns
    n_paths = 1000
    paths = np.zeros((n_paths, n_steps))
    
    for i in range(n_paths):
        # Bootstrap random returns
        random_indices = np.random.randint(0, len(log_returns), n_steps)
        path_returns = log_returns[random_indices]
        
        # Cumulative sum of returns
        cumulative_returns = np.cumsum(path_returns)
        
        # Convert to price
        paths[i] = prices[-1] * np.exp(cumulative_returns)
    
    # Apply quantum weighting to paths
    path_weights = qpl_generator.filter_paths_by_levels(paths)
    
    # Find most probable path
    most_likely_idx = np.argmax(path_weights)
    most_likely_path = paths[most_likely_idx]
    
    # Visualize the results
    fig = qpl_generator.visualize_price_levels(
        prices, 
        paths, 
        dates, 
        path_weights, 
        most_likely_path
    )
    
    # Return comprehensive results
    return {
        'symbol': symbol,
        'prices': prices,
        'dates': dates,
        'quantum_results': {
            'price_levels': qpl_results,
            'multi_dimensional': multi_results
        },
        'paths': paths,
        'path_weights': path_weights,
        'most_likely_path': most_likely_path,
        'figure': fig
    }