"""
Experiment configuration for FracTime scientific research.

This module defines the configuration for running comprehensive model comparisons,
including the test universe of assets, forecast horizons, and experiment parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class AssetConfig:
    """Configuration for a single asset in the test universe."""

    ticker: str
    name: str
    category: str
    subcategory: str
    frequency: str = "daily"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __post_init__(self) -> None:
        if self.start_date is None:
            if self.category == "crypto":
                self.start_date = "2017-01-01"
            else:
                self.start_date = "2015-01-01"
        if self.end_date is None:
            self.end_date = "2024-12-31"


# Forecast horizons in trading days
HORIZONS: Dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "21d": 21,
    "63d": 63,
}

# Complete test universe
TEST_UNIVERSE: List[AssetConfig] = [
    # Equities - Tech
    AssetConfig("AAPL", "Apple Inc.", "equity", "tech"),
    AssetConfig("MSFT", "Microsoft Corp.", "equity", "tech"),
    AssetConfig("GOOGL", "Alphabet Inc.", "equity", "tech"),
    AssetConfig("NVDA", "NVIDIA Corp.", "equity", "tech"),
    AssetConfig("META", "Meta Platforms", "equity", "tech"),
    # Equities - Finance
    AssetConfig("JPM", "JPMorgan Chase", "equity", "finance"),
    AssetConfig("GS", "Goldman Sachs", "equity", "finance"),
    AssetConfig("BAC", "Bank of America", "equity", "finance"),
    # Equities - Healthcare
    AssetConfig("JNJ", "Johnson & Johnson", "equity", "healthcare"),
    AssetConfig("UNH", "UnitedHealth Group", "equity", "healthcare"),
    # Equities - Energy
    AssetConfig("XOM", "Exxon Mobil", "equity", "energy"),
    AssetConfig("CVX", "Chevron Corp.", "equity", "energy"),
    # Equities - Consumer
    AssetConfig("WMT", "Walmart Inc.", "equity", "consumer"),
    AssetConfig("PG", "Procter & Gamble", "equity", "consumer"),
    AssetConfig("KO", "Coca-Cola Co.", "equity", "consumer"),
    # Indices & ETFs
    AssetConfig("^GSPC", "S&P 500", "index", "broad"),
    AssetConfig("^DJI", "Dow Jones Industrial", "index", "broad"),
    AssetConfig("^IXIC", "NASDAQ Composite", "index", "broad"),
    AssetConfig("^VIX", "CBOE Volatility Index", "index", "volatility"),
    AssetConfig("XLF", "Financial Select SPDR", "etf", "sector"),
    AssetConfig("XLK", "Technology Select SPDR", "etf", "sector"),
    AssetConfig("XLE", "Energy Select SPDR", "etf", "sector"),
    AssetConfig("TLT", "iShares 20+ Year Treasury", "etf", "bonds"),
    AssetConfig("GLD", "SPDR Gold Shares", "etf", "commodity"),
    AssetConfig("USO", "US Oil Fund", "etf", "commodity"),
    # Crypto (daily)
    AssetConfig("BTC-USD", "Bitcoin", "crypto", "major"),
    AssetConfig("ETH-USD", "Ethereum", "crypto", "major"),
    AssetConfig("SOL-USD", "Solana", "crypto", "alt", start_date="2020-04-01"),
    AssetConfig("ADA-USD", "Cardano", "crypto", "alt", start_date="2017-10-01"),
    # Forex
    AssetConfig("EURUSD=X", "EUR/USD", "forex", "major"),
    AssetConfig("GBPUSD=X", "GBP/USD", "forex", "major"),
    AssetConfig("USDJPY=X", "USD/JPY", "forex", "major"),
    AssetConfig("AUDUSD=X", "AUD/USD", "forex", "major"),
]

# Crypto hourly subset for intraday analysis
CRYPTO_HOURLY: List[AssetConfig] = [
    AssetConfig(
        "BTC-USD",
        "Bitcoin",
        "crypto",
        "major",
        frequency="hourly",
        start_date="2022-01-01",
    ),
    AssetConfig(
        "ETH-USD",
        "Ethereum",
        "crypto",
        "major",
        frequency="hourly",
        start_date="2022-01-01",
    ),
]

# Weekly subset for longer horizon validation
WEEKLY_SUBSET: List[AssetConfig] = [
    AssetConfig(
        "^GSPC",
        "S&P 500",
        "index",
        "broad",
        frequency="weekly",
        start_date="2010-01-01",
    ),
    AssetConfig(
        "BTC-USD",
        "Bitcoin",
        "crypto",
        "major",
        frequency="weekly",
        start_date="2017-01-01",
    ),
]

# Model configurations
MODELS: Dict[str, Dict[str, Any]] = {
    "fractime_rs": {
        "class": "Forecaster",
        "module": "fractime",
        "params": {"method": "rs", "lookback": 252},
    },
    "fractime_rs_v2": {
        "class": "Forecaster",
        "module": "fractime",
        "params": {"method": "rs", "lookback": 252, "scoring": "v2"},
    },
    "fractime_rs_v3": {
        "class": "Forecaster",
        "module": "fractime",
        "params": {"method": "rs", "lookback": 252, "scoring": "v3"},
    },
    "fractime_dfa": {
        "class": "Forecaster",
        "module": "fractime",
        "params": {"method": "dfa", "lookback": 252},
    },
    "fractime_ensemble": {
        "class": "Ensemble",
        "module": "fractime",
        "params": {"strategy": "weighted"},
    },
    "arima": {
        "class": "ARIMAForecaster",
        "module": "fractime.baselines",
        "params": {},
    },
    "garch": {
        "class": "GARCHForecaster",
        "module": "fractime.baselines",
        "params": {},
    },
    "prophet": {
        "class": "ProphetForecaster",
        "module": "fractime.baselines",
        "params": {},
    },
    "ets": {
        "class": "ETSForecaster",
        "module": "fractime.baselines",
        "params": {},
    },
    "lstm": {
        "class": "LSTMForecaster",
        "module": "fractime.baselines",
        "params": {"hidden_size": 64, "num_layers": 2, "epochs": 50},
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for a complete research experiment.

    Attributes:
        name: Unique name for this experiment
        description: Description of the experiment
        assets: List of assets to test
        horizons: Forecast horizons in trading days
        models: List of model names to compare
        train_window: Initial training window size
        step_size: Number of steps between forecasts
        window_type: 'expanding' or 'rolling' window
        rolling_window_size: Size of rolling window (if window_type='rolling')
        n_paths: Number of Monte Carlo paths for probabilistic forecasts
        confidence_levels: Confidence levels for interval evaluation
        output_dir: Directory for saving results
        random_seed: Random seed for reproducibility
    """

    name: str = "default_experiment"
    description: str = "FracTime model comparison study"
    assets: List[AssetConfig] = field(default_factory=lambda: TEST_UNIVERSE.copy())
    horizons: Dict[str, int] = field(default_factory=lambda: HORIZONS.copy())
    models: List[str] = field(
        default_factory=lambda: [
            "fractime_rs",
            "fractime_dfa",
            "arima",
            "garch",
            "prophet",
            "ets",
        ]
    )
    train_window: int = 252
    step_size: int = 21
    window_type: str = "expanding"
    rolling_window_size: Optional[int] = None
    n_paths: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.50, 0.80, 0.95])
    output_dir: Path = field(default_factory=lambda: Path("research_results"))
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "experiments").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "site").mkdir(exist_ok=True)

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Create default experiment configuration."""
        return cls()

    @classmethod
    def quick_test(cls) -> "ExperimentConfig":
        """Create a quick test configuration with reduced assets and horizons."""
        return cls(
            name="quick_test",
            description="Quick test run with reduced universe",
            assets=[
                AssetConfig("^GSPC", "S&P 500", "index", "broad"),
                AssetConfig("AAPL", "Apple Inc.", "equity", "tech"),
                AssetConfig("BTC-USD", "Bitcoin", "crypto", "major"),
            ],
            horizons={"1d": 1, "5d": 5},
            models=["fractime_rs", "arima", "garch"],
            step_size=63,
            n_paths=500,
        )

    @classmethod
    def full_study(cls) -> "ExperimentConfig":
        """Create full study configuration including LSTM."""
        return cls(
            name="full_study",
            description="Complete FracTime research study",
            models=list(MODELS.keys()),
            step_size=21,
            n_paths=1000,
        )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return MODELS[model_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "assets": [
                {
                    "ticker": a.ticker,
                    "name": a.name,
                    "category": a.category,
                    "subcategory": a.subcategory,
                    "frequency": a.frequency,
                    "start_date": a.start_date,
                    "end_date": a.end_date,
                }
                for a in self.assets
            ],
            "horizons": self.horizons,
            "models": self.models,
            "train_window": self.train_window,
            "step_size": self.step_size,
            "window_type": self.window_type,
            "rolling_window_size": self.rolling_window_size,
            "n_paths": self.n_paths,
            "confidence_levels": self.confidence_levels,
            "output_dir": str(self.output_dir),
            "random_seed": self.random_seed,
            "timestamp": datetime.now().isoformat(),
        }


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig.default()
