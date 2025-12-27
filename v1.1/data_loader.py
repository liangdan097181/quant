import akshare as ak
import pandas as pd
import os
from datetime import datetime

class StockDataFetcher:
    def __init__(self, cache_dir='data'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_us_stock_daily(self, symbol, use_cache=True):
        """
        使用 akshare 获取美股历史行情数据
        symbol: 例如 'AAPL' 或 'IXIC'（注意：指数不需要^前缀）
        注意：akshare 的美股接口通常返回全量历史数据
        """
        # 移除可能的^前缀，AkShare可能不支持
        symbol = symbol.strip('^')
        
        file_path = os.path.join(self.cache_dir, f"us_{symbol}_daily.parquet")
        
        if use_cache and os.path.exists(file_path):
            print(f"Loading {symbol} data from cache...")
            return pd.read_parquet(file_path)
        
        print(f"Downloading {symbol} data from AkShare...")
        try:
            # 使用 stock_us_daily 接口，移除 symbol_kind 参数
            df = ak.stock_us_daily(symbol=symbol)
            if df.empty:
                # 尝试使用指数专用接口
                print(f"Trying index interface for {symbol}...")
                # AkShare 美股指数数据接口可能不同，尝试使用 stock_us_index_daily
                try:
                    df = ak.stock_us_index_daily(symbol=symbol)
                except Exception as e2:
                    print(f"Index interface failed: {e2}")
                    raise ValueError(f"No data found for {symbol}")
            
            # 统一列名以适配回测引擎 (Open, High, Low, Close, Volume)
            # AkShare 返回的列通常是 date, open, high, low, close, volume 等
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
            
            # 重命名列以符合惯例（首字母大写）
            if 'open' in df.columns:
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
            
            # 保存到本地缓存
            df.to_parquet(file_path)
            return df
        except Exception as e:
            print(f"Error downloading {symbol} via AkShare: {e}")
            return None

if __name__ == "__main__":
    # 测试代码
    fetcher = StockDataFetcher()
    # 示例获取苹果公司数据
    data = fetcher.get_us_stock_daily("AAPL")
    if data is not None:
        print("Data Head:")
        print(data.head())
        print("\nColumns:", data.columns.tolist())
