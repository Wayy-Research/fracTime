"""
Other data sources: Forex, Commodities, Bonds.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import polars as pl
import pandas as pd
import requests

from .base import DataSource, DataSourceConfig, TimeSeriesData


class ForexAlphaVantageSource(DataSource):
    """Forex data from Alpha Vantage."""

    @property
    def source_name(self) -> str:
        return "forex_alphavantage"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['forex']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch forex data from Alpha Vantage.

        Args:
            symbol: Currency pair like 'EUR/USD' or 'EURUSD'
        """
        if not self.config.api_key:
            raise ValueError("Alpha Vantage requires an API key")

        from alpha_vantage.foreignexchange import ForeignExchange

        self._enforce_rate_limit()

        def _fetch():
            # Parse currency pair
            if '/' in symbol:
                from_currency, to_currency = symbol.split('/')
            else:
                # Assume format like EURUSD
                from_currency = symbol[:3]
                to_currency = symbol[3:]

            fx = ForeignExchange(key=self.config.api_key, output_format='pandas')
            data, meta = fx.get_currency_exchange_daily(
                from_symbol=from_currency,
                to_symbol=to_currency,
                outputsize='full'
            )

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Reset index
            data = data.reset_index()
            data = data.rename(columns={'date': 'Date'})

            # Rename columns
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
            }
            data = data.rename(columns=column_mapping)
            data['Volume'] = 0.0  # Forex doesn't have volume

            # Filter by date range
            data['Date'] = pd.to_datetime(data['Date'])
            data = data[data['Date'] >= start_date]
            if end_date:
                data = data[data['Date'] <= end_date]

            # Convert to Polars
            pl_df = pl.from_pandas(data)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search forex pairs."""
        # Common forex pairs
        major_pairs = [
            ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'),
            ('USD', 'CHF'), ('AUD', 'USD'), ('USD', 'CAD'),
            ('NZD', 'USD'), ('EUR', 'GBP'), ('EUR', 'JPY'),
            ('GBP', 'JPY')
        ]

        query_upper = query.upper().replace('/', '')
        results = []

        for base, quote in major_pairs:
            pair = f"{base}{quote}"
            if query_upper in pair:
                results.append({
                    'symbol': f"{base}/{quote}",
                    'name': f"{base}/{quote}",
                    'type': 'forex',
                    'base': base,
                    'quote': quote,
                    'source': self.source_name
                })

                if len(results) >= limit:
                    break

        return results

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get forex pair metadata."""
        if '/' in symbol:
            base, quote = symbol.split('/')
        else:
            base = symbol[:3]
            quote = symbol[3:]

        return {
            'symbol': symbol,
            'base_currency': base,
            'quote_currency': quote,
            'type': 'forex',
            'source': self.source_name
        }


class CommodityAPISource(DataSource):
    """Commodity prices from API Ninjas."""

    @property
    def source_name(self) -> str:
        return "api_ninjas_commodity"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['commodity']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch commodity data from API Ninjas.

        Note: API Ninjas provides current prices. For historical data,
        this would need to be called repeatedly and cached.
        """
        if not self.config.api_key:
            raise ValueError("API Ninjas requires an API key")

        self._enforce_rate_limit()

        def _fetch():
            url = f"https://api.api-ninjas.com/v1/commodityprice?name={symbol}"
            headers = {'X-Api-Key': self.config.api_key}

            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()

            data = response.json()

            if not data:
                raise ValueError(f"No data returned for {symbol}")

            # API Ninjas returns current price only
            # For full historical data, would need caching/database
            price = data.get('price', 0)

            # Create a single-row DataFrame
            df = pd.DataFrame([{
                'Date': datetime.now(),
                'Open': price,
                'High': price,
                'Low': price,
                'Close': price,
                'Volume': 0.0
            }])

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'commodity': symbol,
                    'note': 'Current price only - historical requires caching',
                    'unit': data.get('unit', ''),
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search commodities."""
        # Common commodities
        commodities = [
            'gold', 'silver', 'copper', 'platinum', 'palladium',
            'crude_oil', 'brent_oil', 'natural_gas',
            'wheat', 'corn', 'soybeans', 'coffee', 'sugar', 'cotton'
        ]

        query_lower = query.lower()
        results = []

        for commodity in commodities:
            if query_lower in commodity:
                results.append({
                    'symbol': commodity,
                    'name': commodity.replace('_', ' ').title(),
                    'type': 'commodity',
                    'source': self.source_name
                })

                if len(results) >= limit:
                    break

        return results

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get commodity metadata."""
        return {
            'symbol': symbol,
            'type': 'commodity',
            'note': 'Real-time pricing from CME/NYMEX',
            'source': self.source_name
        }


class USTreasurySource(DataSource):
    """US Treasury bond yields."""

    @property
    def source_name(self) -> str:
        return "us_treasury"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['bond']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch US Treasury yields.

        Symbol should be like 'DGS10' for 10-year treasury (use FRED codes).
        For direct Treasury API, use 'avg_interest_rates' endpoint.
        """
        self._enforce_rate_limit()

        def _fetch():
            # Use Treasury Fiscal Data API
            url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"

            params = {
                'filter': f'record_date:gte:{start_date}',
                'sort': '-record_date',
                'page[size]': 10000
            }

            if end_date:
                params['filter'] += f',record_date:lte:{end_date}'

            response = requests.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()

            data = response.json()
            records = data.get('data', [])

            if not records:
                raise ValueError(f"No data returned for US Treasury rates")

            # Convert to DataFrame
            df = pd.DataFrame(records)

            # Select relevant columns
            df = df.rename(columns={'record_date': 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])

            # Use avg_interest_rate_amt as Close price
            if 'avg_interest_rate_amt' in df.columns:
                df['Close'] = pd.to_numeric(df['avg_interest_rate_amt'], errors='coerce')
            else:
                raise ValueError("Interest rate data not found in response")

            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0.0

            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'instrument': 'US Treasury Average Interest Rates',
                    'note': 'Average interest rates on US Treasury securities',
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search treasury instruments."""
        treasuries = [
            {'symbol': 'DGS1MO', 'name': '1-Month Treasury'},
            {'symbol': 'DGS3MO', 'name': '3-Month Treasury'},
            {'symbol': 'DGS6MO', 'name': '6-Month Treasury'},
            {'symbol': 'DGS1', 'name': '1-Year Treasury'},
            {'symbol': 'DGS2', 'name': '2-Year Treasury'},
            {'symbol': 'DGS5', 'name': '5-Year Treasury'},
            {'symbol': 'DGS10', 'name': '10-Year Treasury'},
            {'symbol': 'DGS20', 'name': '20-Year Treasury'},
            {'symbol': 'DGS30', 'name': '30-Year Treasury'},
        ]

        query_upper = query.upper()
        results = []

        for treasury in treasuries:
            if query_upper in treasury['symbol'] or query_upper in treasury['name'].upper():
                results.append({
                    'symbol': treasury['symbol'],
                    'name': treasury['name'],
                    'type': 'bond',
                    'issuer': 'US Treasury',
                    'source': self.source_name
                })

                if len(results) >= limit:
                    break

        return results if results else treasuries[:limit]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get treasury metadata."""
        return {
            'symbol': symbol,
            'type': 'bond',
            'issuer': 'US Treasury',
            'note': 'Daily treasury yield rates',
            'source': self.source_name
        }
