"""
Cryptocurrency data sources.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import polars as pl
import pandas as pd

from .base import DataSource, DataSourceConfig, TimeSeriesData


class BinanceSource(DataSource):
    """Binance cryptocurrency exchange data source."""

    @property
    def source_name(self) -> str:
        return "binance"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['crypto']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch crypto data from Binance."""
        from binance.client import Client

        self._enforce_rate_limit()

        def _fetch():
            client = Client()  # Public API, no key needed

            # Convert interval to Binance format
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK,
                '1mo': Client.KLINE_INTERVAL_1MONTH,
            }
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1DAY)

            # Fetch historical klines
            klines = client.get_historical_klines(
                symbol=symbol.upper(),
                interval=binance_interval,
                start_str=start_date,
                end_str=end_date
            )

            if not klines:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Select and convert columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
                df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'interval': interval,
                    'exchange': 'Binance',
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
        """Search for crypto trading pairs on Binance."""
        from binance.client import Client

        self._enforce_rate_limit()

        try:
            client = Client()
            exchange_info = client.get_exchange_info()

            query_upper = query.upper()
            results = []

            for symbol_info in exchange_info['symbols']:
                if query_upper in symbol_info['symbol']:
                    results.append({
                        'symbol': symbol_info['symbol'],
                        'name': f"{symbol_info['baseAsset']}/{symbol_info['quoteAsset']}",
                        'type': 'crypto',
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'source': self.source_name
                    })

                    if len(results) >= limit:
                        break

            return results
        except Exception as e:
            print(f"Binance search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a crypto trading pair."""
        from binance.client import Client

        self._enforce_rate_limit()

        try:
            client = Client()
            ticker = client.get_ticker(symbol=symbol.upper())
            exchange_info = client.get_exchange_info()

            # Find symbol info
            symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == symbol.upper()),
                None
            )

            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'volume_24h': float(ticker['volume']),
                'price_change_24h': float(ticker['priceChange']),
                'price_change_percent_24h': float(ticker['priceChangePercent']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'baseAsset': symbol_info['baseAsset'] if symbol_info else '',
                'quoteAsset': symbol_info['quoteAsset'] if symbol_info else '',
                'source': self.source_name
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }


class CoinGeckoSource(DataSource):
    """CoinGecko cryptocurrency data source."""

    @property
    def source_name(self) -> str:
        return "coingecko"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['crypto']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch crypto data from CoinGecko."""
        from pycoingecko import CoinGeckoAPI

        self._enforce_rate_limit()

        def _fetch():
            cg = CoinGeckoAPI()

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            if end_date:
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            else:
                end_ts = int(datetime.now().timestamp())

            # Get market chart data
            # symbol should be coin id (e.g., 'bitcoin', not 'BTC')
            coin_id = symbol.lower()
            data = cg.get_coin_market_chart_range_by_id(
                id=coin_id,
                vs_currency='usd',
                from_timestamp=start_ts,
                to_timestamp=end_ts
            )

            if not data or 'prices' not in data:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [])

            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add volume if available
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
                df = df.merge(vol_df, on='timestamp', how='left')

            # CoinGecko doesn't provide OHLC in free tier, only close prices
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']

            # Select columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'interval': interval,
                    'coin_id': coin_id,
                    'note': 'CoinGecko free tier provides close prices only',
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
        """Search for cryptocurrencies on CoinGecko."""
        from pycoingecko import CoinGeckoAPI

        self._enforce_rate_limit()

        try:
            cg = CoinGeckoAPI()
            results = cg.search(query)

            coins = []
            for coin in results.get('coins', [])[:limit]:
                coins.append({
                    'symbol': coin.get('symbol', '').upper(),
                    'name': coin.get('name', ''),
                    'id': coin.get('id', ''),  # CoinGecko coin ID
                    'type': 'crypto',
                    'market_cap_rank': coin.get('market_cap_rank'),
                    'source': self.source_name
                })

            return coins
        except Exception as e:
            print(f"CoinGecko search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get cryptocurrency metadata from CoinGecko."""
        from pycoingecko import CoinGeckoAPI

        self._enforce_rate_limit()

        try:
            cg = CoinGeckoAPI()
            coin_id = symbol.lower()
            data = cg.get_coin_by_id(coin_id)

            return {
                'symbol': data.get('symbol', '').upper(),
                'name': data.get('name', ''),
                'id': coin_id,
                'description': data.get('description', {}).get('en', '')[:500],
                'market_cap_rank': data.get('market_cap_rank'),
                'market_data': {
                    'current_price': data.get('market_data', {}).get('current_price', {}).get('usd'),
                    'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd'),
                    'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd'),
                    'high_24h': data.get('market_data', {}).get('high_24h', {}).get('usd'),
                    'low_24h': data.get('market_data', {}).get('low_24h', {}).get('usd'),
                },
                'source': self.source_name
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }


class KrakenSource(DataSource):
    """Kraken cryptocurrency exchange data source."""

    @property
    def source_name(self) -> str:
        return "kraken"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['crypto']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch crypto data from Kraken."""
        import krakenex

        self._enforce_rate_limit()

        def _fetch():
            k = krakenex.API()

            # Convert interval to Kraken format (minutes)
            interval_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080,
            }
            kraken_interval = interval_map.get(interval, 1440)

            # Convert start date to timestamp
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())

            # Get OHLC data
            response = k.query_public('OHLC', {
                'pair': symbol.upper(),
                'interval': kraken_interval,
                'since': since
            })

            if response.get('error'):
                raise ValueError(f"Kraken API error: {response['error']}")

            # Get the data for the pair
            pair_key = list(response['result'].keys())[0]
            if pair_key == 'last':
                raise ValueError(f"No data returned for {symbol}")

            ohlc_data = response['result'][pair_key]

            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close',
                'vwap', 'Volume', 'count'
            ])

            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')

            # Select and convert columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
                df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

            # Filter by end date if provided
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df['Date'] <= end_dt]

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'interval': interval,
                    'pair': pair_key,
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
        """Search for trading pairs on Kraken."""
        import krakenex

        self._enforce_rate_limit()

        try:
            k = krakenex.API()
            response = k.query_public('AssetPairs')

            if response.get('error'):
                return []

            pairs = response.get('result', {})
            query_upper = query.upper()
            results = []

            for pair_name, pair_info in pairs.items():
                if query_upper in pair_name or query_upper in pair_info.get('altname', ''):
                    results.append({
                        'symbol': pair_name,
                        'name': pair_info.get('altname', pair_name),
                        'type': 'crypto',
                        'base': pair_info.get('base'),
                        'quote': pair_info.get('quote'),
                        'source': self.source_name
                    })

                    if len(results) >= limit:
                        break

            return results
        except Exception as e:
            print(f"Kraken search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a crypto trading pair."""
        import krakenex

        self._enforce_rate_limit()

        try:
            k = krakenex.API()
            ticker_response = k.query_public('Ticker', {'pair': symbol.upper()})

            if ticker_response.get('error'):
                raise ValueError(ticker_response['error'])

            pair_key = list(ticker_response['result'].keys())[0]
            ticker = ticker_response['result'][pair_key]

            return {
                'symbol': symbol,
                'pair': pair_key,
                'last_price': float(ticker['c'][0]),
                'volume_24h': float(ticker['v'][1]),
                'high_24h': float(ticker['h'][1]),
                'low_24h': float(ticker['l'][1]),
                'trades_24h': ticker['t'][1],
                'source': self.source_name
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }
