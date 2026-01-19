"""
Data Source Registry - Central hub for all data sources with fallback support.
"""

from typing import Dict, List, Optional, Type
import os
from .base import DataSource, DataSourceConfig, TimeSeriesData

# Import all data source implementations
from .equities import YFinanceSource, AlphaVantageSource, TwelveDataSource
from .crypto import BinanceSource, CoinGeckoSource, KrakenSource
from .economic import FREDSource, WorldBankSource, ECBSource
from .other import ForexAlphaVantageSource, CommodityAPISource, USTreasurySource


class DataSourceRegistry:
    """
    Registry for managing multiple data sources with fallback support.
    """

    def __init__(self):
        """Initialize the registry with all available sources."""
        self._sources: Dict[str, Type[DataSource]] = {}
        self._instances: Dict[str, DataSource] = {}
        self._fallback_chains: Dict[str, List[str]] = {}

        # Register all sources
        self._register_sources()
        self._setup_fallback_chains()

    def _register_sources(self) -> None:
        """Register all available data source classes."""
        # Equities
        self._sources['yahoo_finance'] = YFinanceSource
        self._sources['alpha_vantage'] = AlphaVantageSource
        self._sources['twelve_data'] = TwelveDataSource

        # Crypto
        self._sources['binance'] = BinanceSource
        self._sources['coingecko'] = CoinGeckoSource
        self._sources['kraken'] = KrakenSource

        # Economic
        self._sources['fred'] = FREDSource
        self._sources['world_bank'] = WorldBankSource
        self._sources['ecb'] = ECBSource

        # Forex, Commodities, Bonds
        self._sources['forex_alphavantage'] = ForexAlphaVantageSource
        self._sources['api_ninjas_commodity'] = CommodityAPISource
        self._sources['us_treasury'] = USTreasurySource

    def _setup_fallback_chains(self) -> None:
        """
        Setup fallback chains for each asset type.

        If primary source fails, try the next in the chain.
        """
        self._fallback_chains = {
            'equity': ['yahoo_finance', 'twelve_data', 'alpha_vantage'],
            'crypto': ['binance', 'coingecko', 'kraken'],
            'economic': ['fred', 'world_bank'],
            'forex': ['forex_alphavantage'],
            'commodity': ['api_ninjas_commodity', 'alpha_vantage'],
            'bond': ['us_treasury', 'fred'],
        }

    def get_source(self, source_name: str, api_key: Optional[str] = None) -> DataSource:
        """
        Get a data source instance by name.

        Args:
            source_name: Name of the data source
            api_key: Optional API key (will try environment variable if not provided)

        Returns:
            DataSource instance

        Raises:
            ValueError: If source name is not registered
        """
        if source_name not in self._sources:
            raise ValueError(f"Unknown data source: {source_name}")

        # Return cached instance if it exists
        if source_name in self._instances:
            return self._instances[source_name]

        # Try to get API key from environment if not provided
        if api_key is None:
            env_var = f"{source_name.upper()}_API_KEY"
            api_key = os.getenv(env_var)

        # Create configuration
        config = DataSourceConfig(
            name=source_name,
            api_key=api_key
        )

        # Create and cache instance
        source_class = self._sources[source_name]
        instance = source_class(config)
        self._instances[source_name] = instance

        return instance

    def get_data_with_fallback(
        self,
        asset_type: str,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        preferred_source: Optional[str] = None,
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch data with automatic fallback to alternative sources.

        Args:
            asset_type: Type of asset ('equity', 'crypto', 'forex', etc.)
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date (optional)
            interval: Data interval
            preferred_source: Preferred source name (optional)
            **kwargs: Additional parameters passed to data source

        Returns:
            TimeSeriesData from first successful source

        Raises:
            RuntimeError: If all sources in the fallback chain fail
        """
        # Get fallback chain for this asset type
        chain = self._fallback_chains.get(asset_type, [])

        if not chain:
            raise ValueError(f"No data sources available for asset type: {asset_type}")

        # Put preferred source first if specified
        if preferred_source and preferred_source in chain:
            chain = [preferred_source] + [s for s in chain if s != preferred_source]

        errors = {}

        # Try each source in the chain
        for source_name in chain:
            try:
                source = self.get_source(source_name)
                print(f"Trying {source_name} for {symbol}...")

                data = source.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    **kwargs
                )

                print(f"✓ Successfully fetched data from {source_name}")
                return data

            except Exception as e:
                errors[source_name] = str(e)
                print(f"✗ {source_name} failed: {e}")
                continue

        # All sources failed
        error_msg = "All data sources failed:\n"
        for source, error in errors.items():
            error_msg += f"  - {source}: {error}\n"

        raise RuntimeError(error_msg)

    def list_sources(
        self,
        asset_type: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        List all registered data sources.

        Args:
            asset_type: Filter by asset type (optional)

        Returns:
            List of source information dictionaries
        """
        result = []

        for source_name, source_class in self._sources.items():
            # Create temporary instance to get metadata
            config = DataSourceConfig(name=source_name)
            instance = source_class(config)

            supported_types = instance.supported_asset_types

            # Filter by asset type if specified
            if asset_type and asset_type not in supported_types:
                continue

            result.append({
                'name': source_name,
                'supported_types': supported_types,
                'requires_api_key': source_name in [
                    'alpha_vantage', 'twelve_data', 'fred',
                    'forex_alphavantage', 'api_ninjas_commodity'
                ]
            })

        return result

    def search_symbols_across_sources(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit_per_source: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search for symbols across all applicable sources.

        Args:
            query: Search query
            asset_type: Filter by asset type (optional)
            limit_per_source: Max results per source

        Returns:
            Dictionary mapping source names to search results
        """
        results = {}

        for source_name in self._sources.keys():
            try:
                source = self.get_source(source_name)

                # Skip if asset type doesn't match
                if asset_type and asset_type not in source.supported_asset_types:
                    continue

                search_results = source.search_symbols(
                    query=query,
                    asset_type=asset_type,
                    limit=limit_per_source
                )

                if search_results:
                    results[source_name] = search_results

            except Exception as e:
                print(f"Search failed for {source_name}: {e}")
                continue

        return results


# Global registry instance
_registry = DataSourceRegistry()


def get_data_source(source_name: str, api_key: Optional[str] = None) -> DataSource:
    """
    Get a data source instance from the global registry.

    Args:
        source_name: Name of the data source
        api_key: Optional API key

    Returns:
        DataSource instance
    """
    return _registry.get_source(source_name, api_key)


def list_sources(asset_type: Optional[str] = None) -> List[Dict[str, any]]:
    """
    List all available data sources.

    Args:
        asset_type: Filter by asset type (optional)

    Returns:
        List of source information
    """
    return _registry.list_sources(asset_type)


def get_data(
    source_name: str,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = '1d',
    **kwargs
) -> TimeSeriesData:
    """
    Fetch data from a specific source.

    Args:
        source_name: Name of the data source
        symbol: Symbol to fetch
        start_date: Start date
        end_date: End date (optional)
        interval: Data interval
        **kwargs: Additional parameters

    Returns:
        TimeSeriesData
    """
    source = _registry.get_source(source_name)
    return source.get_data(symbol, start_date, end_date, interval, **kwargs)


def get_data_with_fallback(
    asset_type: str,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = '1d',
    **kwargs
) -> TimeSeriesData:
    """
    Fetch data with automatic fallback.

    Args:
        asset_type: Type of asset
        symbol: Symbol to fetch
        start_date: Start date
        end_date: End date (optional)
        interval: Data interval
        **kwargs: Additional parameters

    Returns:
        TimeSeriesData from first successful source
    """
    return _registry.get_data_with_fallback(
        asset_type, symbol, start_date, end_date, interval, **kwargs
    )
