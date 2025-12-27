import pandas as pd
import numpy as np
from scipy.optimize import minimize

class VectorizedBacktester:
    """
    轻量级向量化回测引擎
    """
    def __init__(self, initial_capital=100000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission # 默认单边千分之一

    def run(self, df, signal_column=None, factor_weights=None):
        """
        运行回测
        df: 包含价格和信号的 DataFrame
        signal_column: 信号列名 (1: 做多, -1: 做空, 0: 空仓)
        factor_weights: 因子权重配置，格式为 {"factor_name": weight}
        """
        data = df.copy()
        
        # 如果没有指定信号列，使用因子加权生成信号
        if signal_column is None:
            data['signal'] = self.calculate_combined_signal(data, factor_weights)
            signal_column = 'signal'
        
        # 计算对数收益率
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # 信号对齐（今天产生的信号，明天开盘/收盘交易，这里简化为次日收益）
        data['strategy_returns'] = data[signal_column].shift(1) * data['returns']
        
        # 简单处理手续费 (当信号改变时扣除)
        data['trades'] = data[signal_column].diff().fillna(0).abs()
        data['strategy_returns'] -= data['trades'] * self.commission
        
        # 计算累计收益
        data['cum_market_returns'] = data['returns'].cumsum().apply(np.exp)
        data['cum_strategy_returns'] = data['strategy_returns'].cumsum().apply(np.exp)
        
        # 计算净值 (Wealth Index)
        data['equity'] = data['cum_strategy_returns'] * self.initial_capital
        
        # 记录交易信号和价格
        data['position'] = data[signal_column].shift(1).fillna(0)
        data['prev_position'] = data['position'].shift(1).fillna(0)
        
        # 标记买入和卖出信号
        data['buy_signal'] = (data['position'] == 1) & (data['prev_position'] != 1)
        data['sell_signal'] = (data['position'] == 0) & (data['prev_position'] == 1)
        
        # 记录买入和卖出价格
        data['buy_price'] = 0.0
        data['sell_price'] = 0.0
        data.loc[data['buy_signal'], 'buy_price'] = data['Close']
        data.loc[data['sell_signal'], 'sell_price'] = data['Close']
        
        # 计算统计指标
        perf = self.calculate_performance(data)
        
        return data, perf
    
    def calculate_combined_signal(self, df, factor_weights):
        """
        计算组合信号：根据因子权重加权生成最终信号
        df: 包含因子数据的 DataFrame
        factor_weights: 因子权重配置
        """
        if factor_weights is None:
            factor_weights = {
                'momentum': 0.3,
                'rsi': 0.2,
                'macd': 0.2,
                'liquidity_factor': 0.1,
                'crowding_factor': 0.1,
                'dynamic_momentum_weight': 0.1
            }
        
        # 初始化信号
        signal = pd.Series(0.0, index=df.index)
        
        # 计算各个因子的信号贡献
        for factor_name, weight in factor_weights.items():
            if factor_name in df.columns:
                # 标准化因子值到 [-1, 1] 范围
                factor_values = df[factor_name]
                # 移除极值
                factor_values = self.winsorize(factor_values)
                # 标准化
                factor_values = self.standardize(factor_values)
                # 添加到信号中
                signal += factor_values * weight
        
        # 应用风险控制
        signal = self.apply_risk_controls(df, signal)
        
        # 转换为离散信号 (1: 做多, 0: 空仓)
        # 这里简化处理，实际应该根据阈值确定
        final_signal = signal.apply(lambda x: 1 if x > 0 else 0)
        
        return final_signal
    
    def winsorize(self, series, lower=0.05, upper=0.95):
        """
        极值处理：将超过分位数的值替换为分位数值
        """
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def standardize(self, series):
        """
        标准化：将序列转换为均值为0，标准差为1
        """
        return (series - series.mean()) / (series.std() + 1e-8)  # 避免除以0
    
    def apply_risk_controls(self, df, signal):
        """
        应用风险控制措施
        """
        # 1. 波动率控制：当波动率过高时降低仓位
        if 'bayesian_volatility' in df.columns:
            vol_threshold = df['bayesian_volatility'].quantile(0.75)
            signal = signal * (1 - df['bayesian_volatility'].apply(lambda x: 0.5 if x > vol_threshold else 0))
        
        # 2. 流动性控制：当流动性不足时降低仓位
        if 'liquidity_factor' in df.columns:
            liq_threshold = df['liquidity_factor'].quantile(0.75)
            signal = signal * (1 - df['liquidity_factor'].apply(lambda x: 0.5 if x > liq_threshold else 0))
        
        # 3. 拥挤度控制：当因子拥挤时降低仓位
        if 'crowding_factor' in df.columns:
            crowd_threshold = df['crowding_factor'].quantile(0.8)
            signal = signal * (1 - df['crowding_factor'].apply(lambda x: 0.5 if x > crowd_threshold else 0))
        
        return signal
    
    def get_trade_records(self, data, ticker):
        """
        生成详细的交易记录
        data: 回测结果数据
        ticker: 股票代码
        """
        trades = []
        trade_id = 0
        
        # 找到所有的买入信号
        buy_dates = data[data['buy_signal']].index
        sell_dates = data[data['sell_signal']].index
        
        # 确保买入和卖出信号数量匹配
        buy_count = len(buy_dates)
        sell_count = len(sell_dates)
        
        # 如果最后一个信号是买入，添加一个卖出信号（回测结束时平仓）
        if buy_count > sell_count:
            last_date = data.index[-1]
            sell_dates = sell_dates.append(pd.Index([last_date]))
        
        # 生成交易记录
        for i in range(min(buy_count, len(sell_dates))):
            buy_date = buy_dates[i]
            sell_date = sell_dates[i]
            
            # 确保卖出日期在买入日期之后
            if sell_date <= buy_date:
                continue
                
            trade_id += 1
            
            # 获取买入和卖出价格
            buy_price = data.loc[buy_date, 'Close']
            sell_price = data.loc[sell_date, 'Close']
            
            # 计算买入股数（使用当时的可用资金，满仓操作）
            # 这里简化处理，使用初始资金，实际应该使用当时的资产
            shares = self.initial_capital // buy_price
            
            # 计算收益
            cost = buy_price * shares
            proceeds = sell_price * shares
            commission = (cost + proceeds) * self.commission
            profit = proceeds - cost - commission
            return_pct = (profit / cost) * 100
            
            # 添加到交易记录
            trades.append({
                '交易编号': trade_id,
                '股票代码': ticker,
                '买入时间': buy_date.strftime('%Y-%m-%d'),
                '买入价格': f"${buy_price:.2f}",
                '买入股数': f"{shares:,}",
                '平仓时间': sell_date.strftime('%Y-%m-%d'),
                '平仓价格': f"${sell_price:.2f}",
                '收益金额': f"${profit:.2f}",
                '收益率': f"{return_pct:.2f}%"
            })
        
        return trades

    def calculate_performance(self, data):
        """
        计算绩效指标
        """
        # 累计收益率
        total_return = data['cum_strategy_returns'].iloc[-1] - 1
        
        # 年化收益率 (假设一年 252 个交易日)
        annual_return = (data['cum_strategy_returns'].iloc[-1] ** (252 / len(data))) - 1
        
        # 年化波动率
        annual_vol = data['strategy_returns'].std() * np.sqrt(252)
        
        # 夏普比率 (假设无风险利率为 0)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        # 最大回撤
        data['cum_max'] = data['equity'].cummax()
        data['drawdown'] = (data['equity'] - data['cum_max']) / data['cum_max']
        max_drawdown = data['drawdown'].min()
        
        return {
            'Total Return (%)': total_return * 100,
            'Annual Return (%)': annual_return * 100,
            'Annual Volatility (%)': annual_vol * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100
        }
    
    def optimize_weights(self, df, factor_list, risk_aversion=1.0):
        """
        优化因子权重，平衡风险和收益
        df: 包含因子数据的 DataFrame
        factor_list: 要优化的因子列表
        risk_aversion: 风险厌恶系数，值越大越保守（默认为1.0）
        """
        # 1. 准备数据
        data = df.copy()
        
        # 2. 定义优化目标函数 - 基于回测结果的夏普比率
        def objective(weights):
            # 将权重转换为字典
            weight_dict = {factor: weight for factor, weight in zip(factor_list, weights)}
            
            # 运行回测获取性能指标
            bt = VectorizedBacktester(initial_capital=self.initial_capital, commission=self.commission)
            backtest_result, metrics = bt.run(data, factor_weights=weight_dict)
            
            # 获取夏普比率（值越大越好）
            sharpe_ratio = metrics['Sharpe Ratio']
            
            # 考虑风险厌恶：调整目标函数，平衡收益和风险
            # 当风险厌恶系数较大时，更注重降低风险
            adjusted_score = sharpe_ratio - risk_aversion * (metrics['Annual Volatility (%)'] / 100)
            
            # 返回负值，因为我们使用的是minimize函数
            return -adjusted_score
        
        # 3. 定义约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: x}  # 权重非负
        ]
        
        # 4. 定义权重边界（0到1）
        bounds = tuple((0, 1) for _ in range(len(factor_list)))
        
        # 5. 初始权重（相等权重）
        initial_weights = np.array([1/len(factor_list)] * len(factor_list))
        
        # 6. 运行优化
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # 7. 提取优化后的权重
        optimized_weights = result.x
        
        # 8. 转换为字典格式
        weights_dict = {factor: weight for factor, weight in zip(factor_list, optimized_weights)}
        
        # 9. 计算最佳得分（夏普比率）
        best_score = -result.fun
        
        return weights_dict, best_score

if __name__ == "__main__":
    # 测试代码
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100
    }, index=dates)
    
    # 一个简单的策略：价格 > 均线 则做多
    df['ma'] = df['Close'].rolling(window=10).mean()
    df['signal'] = 0
    df.loc[df['Close'] > df['ma'], 'signal'] = 1
    
    backtester = VectorizedBacktester()
    results, metrics = backtester.run(df, 'signal')
    print("Metrics:", metrics)
    print(results[['Close', 'signal', 'equity']].tail())
