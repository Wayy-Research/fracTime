"""
Base classes for data source abstraction.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import polars as pl


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    name: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_backoff: float = 2.0  # exponential backoff multiplier


@dataclass
class TimeSeriesData:
    """Standardized time series data structure."""

    symbol: str
    source: str
    data: pl.DataFrame  # Expected columns: Date, Open, High, Low, Close, Volume
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate the data structure."""
        required_columns = {'Date', 'Close'}
        if not required_columns.issubset(set(self.data.columns)):
            raise ValueError(f"Data must contain at least columns: {required_columns}")


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: DataSourceConfig):
        """
        Initialize data source with configuration.

        Args:
            config: Data source configuration
        """
        self.config = config
        self._last_request_time: Optional[datetime] = None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass

    @property
    @abstractmethod
    def supported_asset_types(self) -> List[str]:
        """
        Return list of supported asset types.

        Examples: ['equity', 'crypto', 'forex', 'commodity', 'bond', 'economic']
        """
        pass

    @abstractmethod
    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch time series data for a given symbol.

        Args:
            symbol: Ticker symbol or identifier
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Data interval (e.g., '1d', '1h', '1m')
            **kwargs: Additional source-specific parameters

        Returns:
            TimeSeriesData object with standardized format

        Raises:
            ValueError: If symbol or parameters are invalid
            ConnectionError: If API request fails
            RuntimeError: If data cannot be retrieved
        """
        pass

    @abstractmethod
    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for symbols matching a query.

        Args:
            query: Search query string
            asset_type: Filter by asset type (optional)
            limit: Maximum number of results

        Returns:
            List of dictionaries with symbol information:
            [{'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'equity'}, ...]
        """
        pass

    @abstractmethod
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata about a symbol.

        Args:
            symbol: Ticker symbol or identifier

        Returns:
            Dictionary with symbol metadata
        """
        pass

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            min_interval = 60.0 / self.config.rate_limit
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < min_interval:
                import time
                time.sleep(min_interval - elapsed)

        self._last_request_time = datetime.now()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        import time

        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise

                wait_time = self.config.retry_backoff ** attempt
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
