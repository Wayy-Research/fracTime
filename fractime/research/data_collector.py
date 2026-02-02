"""
Unified data collection for FracTime research.

This module provides data fetching via yfinance with local caching
for reproducibility and efficiency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import polars as pl

try:
    import yfinance as yf
except ImportError:
    yf = None

from .config import AssetConfig, ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class AssetData:
    """Container for asset price data."""

    ticker: str
    name: str
    category: str
    subcategory: str
    frequency: str
    dates: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    volume: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    low: Optional[np.ndarray] = None
    open: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.prices)

    @property
    def start_date(self) -> datetime:
        return self.dates[0]

    @property
    def end_date(self) -> datetime:
        return self.dates[-1]

    def to_frame(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        data = {
            "date": self.dates,
            "price": self.prices,
            "returns": self.returns,
        }
        if self.volume is not None:
            data["volume"] = self.volume
        if self.high is not None:
            data["high"] = self.high
        if self.low is not None:
            data["low"] = self.low
        if self.open is not None:
            data["open"] = self.open
        return pl.DataFrame(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "frequency": self.frequency,
            "dates": [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in self.dates],
            "prices": self.prices.tolist(),
            "returns": self.returns.tolist(),
            "volume": self.volume.tolist() if self.volume is not None else None,
            "high": self.high.tolist() if self.high is not None else None,
            "low": self.low.tolist() if self.low is not None else None,
            "open": self.open.tolist() if self.open is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetData":
        """Create from dictionary."""
        return cls(
            ticker=data["ticker"],
            name=data["name"],
            category=data["category"],
            subcategory=data["subcategory"],
            frequency=data["frequency"],
            dates=np.array([datetime.fromisoformat(d) for d in data["dates"]]),
            prices=np.array(data["prices"]),
            returns=np.array(data["returns"]),
            volume=np.array(data["volume"]) if data.get("volume") else None,
            high=np.array(data["high"]) if data.get("high") else None,
            low=np.array(data["low"]) if data.get("low") else None,
            open=np.array(data["open"]) if data.get("open") else None,
            metadata=data.get("metadata"),
        )


class DataCollector:
    """Unified data collection with caching.

    This class fetches price data from yfinance and caches it locally
    for reproducibility and efficiency.

    Example:
        >>> config = ExperimentConfig.quick_test()
        >>> collector = DataCollector(config)
        >>> data = collector.fetch_all()
        >>> print(data['AAPL'].prices[:5])
    """

    def __init__(
        self,
        config: ExperimentConfig,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> None:
        """Initialize the data collector.

        Args:
            config: Experiment configuration
            cache_dir: Directory for caching data (default: config.output_dir/data)
            force_refresh: If True, ignore cache and fetch fresh data
        """
        if yf is None:
            raise ImportError("yfinance is required for data collection. Install with: pip install yfinance")

        self.config = config
        self.cache_dir = cache_dir or config.output_dir / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh
        self._data_cache: Dict[str, AssetData] = {}

    def _get_cache_path(self, asset: AssetConfig) -> Path:
        """Generate cache file path for an asset."""
        cache_key = f"{asset.ticker}_{asset.frequency}_{asset.start_date}_{asset.end_date}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        safe_ticker = asset.ticker.replace("^", "IDX_").replace("=", "_")
        return self.cache_dir / f"{safe_ticker}_{cache_hash}.json"

    def _load_from_cache(self, asset: AssetConfig) -> Optional[AssetData]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(asset)
        if cache_path.exists() and not self.force_refresh:
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded {asset.ticker} from cache")
                return AssetData.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {asset.ticker}: {e}")
        return None

    def _save_to_cache(self, asset: AssetConfig, data: AssetData) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(asset)
        try:
            with open(cache_path, "w") as f:
                json.dump(data.to_dict(), f)
            logger.info(f"Cached {asset.ticker}")
        except Exception as e:
            logger.warning(f"Failed to cache {asset.ticker}: {e}")

    def fetch_asset(self, asset: AssetConfig) -> Optional[AssetData]:
        """Fetch data for a single asset.

        Args:
            asset: Asset configuration

        Returns:
            AssetData object or None if fetch fails
        """
        cached = self._load_from_cache(asset)
        if cached is not None:
            return cached

        logger.info(f"Fetching {asset.ticker} ({asset.name})")

        try:
            ticker = yf.Ticker(asset.ticker)

            interval = "1d"
            if asset.frequency == "hourly":
                interval = "1h"
            elif asset.frequency == "weekly":
                interval = "1wk"

            df = ticker.history(
                start=asset.start_date,
                end=asset.end_date,
                interval=interval,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No data returned for {asset.ticker}")
                return None

            df = df.dropna(subset=["Close"])

            if len(df) < 100:
                logger.warning(f"Insufficient data for {asset.ticker}: {len(df)} rows")
                return None

            dates = df.index.to_numpy()
            prices = df["Close"].to_numpy()
            returns = np.diff(np.log(prices))
            returns = np.concatenate([[0], returns])

            volume = df["Volume"].to_numpy() if "Volume" in df.columns else None
            high = df["High"].to_numpy() if "High" in df.columns else None
            low = df["Low"].to_numpy() if "Low" in df.columns else None
            open_prices = df["Open"].to_numpy() if "Open" in df.columns else None

            metadata = {
                "fetch_time": datetime.now().isoformat(),
                "source": "yfinance",
                "n_rows": len(df),
            }

            data = AssetData(
                ticker=asset.ticker,
                name=asset.name,
                category=asset.category,
                subcategory=asset.subcategory,
                frequency=asset.frequency,
                dates=dates,
                prices=prices,
                returns=returns,
                volume=volume,
                high=high,
                low=low,
                open=open_prices,
                metadata=metadata,
            )

            self._save_to_cache(asset, data)
            return data

        except Exception as e:
            logger.error(f"Failed to fetch {asset.ticker}: {e}")
            return None

    def fetch_all(self, verbose: bool = True) -> Dict[str, AssetData]:
        """Fetch data for all assets in the configuration.

        Args:
            verbose: Print progress information

        Returns:
            Dictionary mapping ticker to AssetData
        """
        results: Dict[str, AssetData] = {}
        failed: List[str] = []

        for i, asset in enumerate(self.config.assets):
            if verbose:
                print(f"[{i + 1}/{len(self.config.assets)}] Fetching {asset.ticker}...")

            data = self.fetch_asset(asset)
            if data is not None:
                key = f"{asset.ticker}_{asset.frequency}" if asset.frequency != "daily" else asset.ticker
                results[key] = data
            else:
                failed.append(asset.ticker)

        if verbose:
            print(f"\nSuccessfully fetched: {len(results)} assets")
            if failed:
                print(f"Failed: {failed}")

        self._data_cache.update(results)
        return results

    def get_cached(self) -> Dict[str, AssetData]:
        """Get all cached data."""
        return self._data_cache

    def get_asset(self, ticker: str) -> Optional[AssetData]:
        """Get a specific asset from cache or fetch if not available."""
        if ticker in self._data_cache:
            return self._data_cache[ticker]

        for asset in self.config.assets:
            if asset.ticker == ticker:
                data = self.fetch_asset(asset)
                if data:
                    self._data_cache[ticker] = data
                return data

        return None

    def summary(self) -> pl.DataFrame:
        """Get summary statistics for all cached data."""
        if not self._data_cache:
            return pl.DataFrame()

        rows = []
        for ticker, data in self._data_cache.items():
            rows.append({
                "ticker": ticker,
                "name": data.name,
                "category": data.category,
                "subcategory": data.subcategory,
                "frequency": data.frequency,
                "n_observations": len(data),
                "start_date": str(data.start_date)[:10],
                "end_date": str(data.end_date)[:10],
                "mean_return": float(np.mean(data.returns)),
                "std_return": float(np.std(data.returns)),
                "min_price": float(np.min(data.prices)),
                "max_price": float(np.max(data.prices)),
            })

        return pl.DataFrame(rows)


def validate_data(data: AssetData) -> Dict[str, Any]:
    """Validate asset data quality.

    Args:
        data: AssetData to validate

    Returns:
        Dictionary with validation results
    """
    results = {
        "ticker": data.ticker,
        "is_valid": True,
        "issues": [],
    }

    if len(data) < 252:
        results["issues"].append(f"Insufficient data: {len(data)} observations")

    if np.isnan(data.prices).any():
        nan_count = np.isnan(data.prices).sum()
        results["issues"].append(f"Contains {nan_count} NaN prices")

    if np.isinf(data.prices).any():
        inf_count = np.isinf(data.prices).sum()
        results["issues"].append(f"Contains {inf_count} infinite prices")

    if (data.prices <= 0).any():
        neg_count = (data.prices <= 0).sum()
        results["issues"].append(f"Contains {neg_count} non-positive prices")

    returns_std = np.std(data.returns[1:])
    if returns_std > 0.5:
        results["issues"].append(f"Unusually high volatility: {returns_std:.4f}")

    extreme_returns = np.abs(data.returns) > 0.5
    if extreme_returns.any():
        count = extreme_returns.sum()
        results["issues"].append(f"Contains {count} extreme returns (>50%)")

    results["is_valid"] = len(results["issues"]) == 0
    return results
