import pandas as pd
import numpy as np

class FactorEngine:
    """
    量化因子计算引擎
    """
    @staticmethod
    def add_sma(df, period=20):
        """简单移动平均线"""
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        return df

    @staticmethod
    def add_rsi(df, period=14):
        """RSI 指标"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """MACD 指标"""
        # 计算EMA
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        # 计算MACD线和信号线
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        df[f'MACD_{fast}_{slow}_{signal}'] = macd
        df[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
        df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
        return df

    @staticmethod
    def add_bollinger_bands(df, length=20, std=2):
        """布林带"""
        sma = df['Close'].rolling(window=length).mean()
        std_dev = df['Close'].rolling(window=length).std()
        
        df[f'BB_Middle_{length}'] = sma
        df[f'BB_Upper_{length}'] = sma + (std_dev * std)
        df[f'BB_Lower_{length}'] = sma - (std_dev * std)
        return df

    # ------------------- 宏观风险因子 -------------------
    @staticmethod
    def add_liquidity_factor(df, window=20):
        """流动性因子：成交量与换手率的监控"""
        # 成交量变化率
        df['volume_change'] = df['Volume'].pct_change()
        # 成交量移动平均
        df['volume_sma'] = df['Volume'].rolling(window=window).mean()
        # 流动性因子：成交量变化率的标准差
        df['liquidity_factor'] = df['volume_change'].rolling(window=window).std()
        return df

    @staticmethod
    def add_volatility_term_structure(df, short_window=20, long_window=60):
        """波动性因子：波动率期限结构"""
        # 计算短期和长期波动率
        df['short_vol'] = df['Close'].pct_change().rolling(window=short_window).std() * np.sqrt(252)
        df['long_vol'] = df['Close'].pct_change().rolling(window=long_window).std() * np.sqrt(252)
        # 波动率期限结构：短期波动率与长期波动率的比率
        df['volatility_term_structure'] = df['short_vol'] / df['long_vol']
        return df

    @staticmethod
    def add_credit_spread_proxy(df, window=20):
        """信用利差代理：使用股价波动性作为代理"""
        # 由于无法直接获取信用利差数据，使用股价波动性作为代理
        # 股价波动性越高，信用风险可能越大
        df['credit_spread_proxy'] = df['Close'].pct_change().rolling(window=window).std() * 100
        return df

    @staticmethod
    def add_inflation_rate_proxy(df, window=60):
        """通胀率代理：使用股价变化率作为代理"""
        # 由于无法直接获取CPI数据，使用股价变化率作为代理
        df['inflation_proxy'] = df['Close'].pct_change().rolling(window=window).mean() * 100
        return df

    # ------------------- 中性因子 -------------------
    @staticmethod
    def add_beta_neutral(df, market_data=None, window=60):
        """Beta中性：计算个股相对于市场的Beta"""
        # 如果没有市场数据，使用自身作为代理
        if market_data is None:
            market_data = df['Close']
        
        # 计算收益率
        stock_returns = df['Close'].pct_change()
        market_returns = market_data.pct_change()
        
        # 计算Beta - 修复：使用不同的列名避免重复标签
        returns_df = pd.DataFrame({
            'stock_returns': stock_returns,
            'market_returns': market_returns
        })
        
        # 计算滚动协方差矩阵
        cov_matrix = returns_df.rolling(window=window).cov()
        
        # 提取股票收益率与市场收益率的协方差和市场收益率的方差
        stock_market_cov = cov_matrix.xs('stock_returns', level=1)['market_returns']
        market_var = cov_matrix.xs('market_returns', level=1)['market_returns']
        
        # 计算Beta
        df['beta'] = stock_market_cov / market_var
        
        # Beta中性调整：计算目标仓位以实现Beta≈0
        df['beta_neutral_signal'] = -df['beta'] * 0.1  # 简化处理
        return df

    @staticmethod
    def add_sector_neutrality(df, sector_weight=0.15):
        """行业中性：在单股票环境下，设置行业权重上限"""
        # 单股票环境下，行业中性主要是限制整体仓位
        df['sector_neutral_signal'] = 1.0  # 基础信号
        # 假设单行业权重上限为15%，简化处理
        df['sector_neutral_signal'] = df['sector_neutral_signal'].clip(upper=sector_weight)
        return df

    @staticmethod
    def add_size_neutrality(df, window=60):
        """市值中性：使用波动率作为市值代理"""
        # 由于无法直接获取市值数据，使用波动率作为代理
        df['size_proxy'] = df['Close'] * df['Volume']
        df['size_neutral_signal'] = 1.0 / (1.0 + df['size_proxy'].pct_change().rolling(window=window).std())
        return df

    # ------------------- 风险预警与防御因子 -------------------
    @staticmethod
    def add_crowding_factor(df, window=20):
        """拥挤度因子：使用成交量和价格变化的相关性"""
        # 计算成交量与价格变化的相关性作为拥挤度指标
        price_change = df['Close'].pct_change()
        volume_change = df['Volume'].pct_change()
        crowding = price_change.rolling(window=window).corr(volume_change)
        df['crowding_factor'] = crowding
        return df

    @staticmethod
    def add_bayesian_shrinkage(df, window=20):
        """特质性波动收缩：贝叶斯法调整波动率估计"""
        # 计算原始波动率
        raw_vol = df['Close'].pct_change().rolling(window=window).std()
        # 计算全局波动率
        global_vol = df['Close'].pct_change().std()
        # 贝叶斯收缩：将原始波动率向全局波动率收缩
        shrinkage = 0.5  # 收缩系数
        df['bayesian_volatility'] = shrinkage * global_vol + (1 - shrinkage) * raw_vol
        return df

    @staticmethod
    def add_dynamic_momentum_reversal(df, window=20, threshold=2.0):
        """动态动能反转因子：当市场出现极端动能时，调整权重"""
        # 计算动量因子
        df['momentum'] = df['Close'].pct_change(window)
        # 计算动量的Z分数
        momentum_mean = df['momentum'].rolling(window=window).mean()
        momentum_std = df['momentum'].rolling(window=window).std()
        df['momentum_zscore'] = (df['momentum'] - momentum_mean) / momentum_std
        
        # 动态调整动能因子权重
        df['dynamic_momentum_weight'] = 1.0
        # 当动量Z分数超过阈值时，降低权重
        df.loc[abs(df['momentum_zscore']) > threshold, 'dynamic_momentum_weight'] = 0.5
        return df

    @classmethod
    def apply_macro_risk_factors(cls, df):
        """应用宏观风险因子"""
        df = cls.add_liquidity_factor(df)
        df = cls.add_volatility_term_structure(df)
        df = cls.add_credit_spread_proxy(df)
        df = cls.add_inflation_rate_proxy(df)
        return df

    @classmethod
    def apply_neutral_factors(cls, df):
        """应用中性因子"""
        df = cls.add_beta_neutral(df)
        df = cls.add_sector_neutrality(df)
        df = cls.add_size_neutrality(df)
        return df

    @classmethod
    def apply_risk_defense_factors(cls, df):
        """应用风险防御因子"""
        df = cls.add_crowding_factor(df)
        df = cls.add_bayesian_shrinkage(df)
        df = cls.add_dynamic_momentum_reversal(df)
        return df

    @classmethod
    def apply_all_factors(cls, df):
        """应用所有因子"""
        df = cls.apply_basic_factors(df)
        df = cls.apply_macro_risk_factors(df)
        df = cls.apply_neutral_factors(df)
        df = cls.apply_risk_defense_factors(df)
        return df

    @classmethod
    def apply_basic_factors(cls, df):
        """应用一组常用基础因子"""
        df = cls.add_sma(df, 20)
        df = cls.add_sma(df, 50)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        return df

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', periods=200)
    df = pd.DataFrame({
        'Close': np.random.randn(200).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, size=200)
    }, index=dates)
    
    engine = FactorEngine()
    result = engine.apply_all_factors(df)
    print("因子计算完成，包含以下列：")
    print(result.columns.tolist())
    print("\n最近5行数据：")
    print(result.tail())
