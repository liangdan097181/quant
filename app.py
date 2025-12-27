import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import uuid

from data_loader import StockDataFetcher
from factor_engine import FactorEngine
from backtest_engine import VectorizedBacktester

# 设置页面
st.set_page_config(page_title="US Quant Pro - 美股量化系统", layout="wide")

st.title("📈 美股量化回测系统 (Alpha v1.1)")
st.markdown("基于 AkShare 数据源的轻量级量化分析工具，支持策略监控和邮箱订阅")

# 初始化session_state存储监控列表
if 'monitor_list' not in st.session_state:
    st.session_state.monitor_list = []

# 生成唯一监控ID
if 'monitor_id_counter' not in st.session_state:
    st.session_state.monitor_id_counter = 1

# 侧边栏：配置参数
st.sidebar.header("策略配置")
ticker = st.sidebar.text_input("股票代码 (如 AAPL, TSLA)", value=".IXIC")
# 设置合理的默认日期范围，方便选择年和月
start_date = st.sidebar.date_input("开始日期", value=datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("结束日期", value=datetime.today())

st.sidebar.divider()
initial_capital = st.sidebar.number_input("初始资金 ($)", value=100000)
commission = st.sidebar.slider("交易手续费 (%)", 0.0, 1.0, 0.1) / 100

# ------------------- 1. 宏观风险因子配置 -------------------
st.sidebar.divider()
st.sidebar.subheader("📊 宏观风险因子")
st.sidebar.markdown("选择1-3种因子组合")

# 流动性因子
use_liquidity = st.sidebar.checkbox("使用流动性因子", value=True, key="use_liquidity")
if use_liquidity:
    liquidity_window = st.sidebar.slider("流动性因子窗口", 10, 60, 20, key="liquidity_window")
    liquidity_weight = st.sidebar.slider("流动性因子权重", 0.0, 1.0, 0.25, step=0.05, key="liquidity_weight")

# 波动性因子
use_volatility = st.sidebar.checkbox("使用波动性因子", value=True, key="use_volatility")
if use_volatility:
    short_vol_window = st.sidebar.slider("短期波动率窗口", 5, 30, 20, key="short_vol_window")
    long_vol_window = st.sidebar.slider("长期波动率窗口", 30, 120, 60, key="long_vol_window")
    volatility_weight = st.sidebar.slider("波动性因子权重", 0.0, 1.0, 0.25, step=0.05, key="volatility_weight")

# 信用利差因子
use_credit_spread = st.sidebar.checkbox("使用信用利差因子", value=False, key="use_credit_spread")
if use_credit_spread:
    credit_spread_window = st.sidebar.slider("信用利差代理窗口", 10, 60, 20, key="credit_spread_window")
    credit_spread_weight = st.sidebar.slider("信用利差因子权重", 0.0, 1.0, 0.25, step=0.05, key="credit_spread_weight")

# 通胀率因子
use_inflation = st.sidebar.checkbox("使用通胀率因子", value=False, key="use_inflation")
if use_inflation:
    inflation_window = st.sidebar.slider("通胀率代理窗口", 30, 120, 60, key="inflation_window")
    inflation_weight = st.sidebar.slider("通胀率因子权重", 0.0, 1.0, 0.25, step=0.05, key="inflation_weight")

# ------------------- 2. 中性因子配置 -------------------
st.sidebar.divider()
st.sidebar.subheader("⚖️ 中性因子")
st.sidebar.markdown("选择1-3种因子组合")

# Beta中性
use_beta = st.sidebar.checkbox("使用Beta中性因子", value=True, key="use_beta")
if use_beta:
    beta_window = st.sidebar.slider("Beta计算窗口", 30, 120, 60, key="beta_window")
    beta_weight = st.sidebar.slider("Beta中性权重", 0.0, 1.0, 0.33, step=0.05, key="beta_weight")

# 行业中性
use_sector = st.sidebar.checkbox("使用行业中性因子", value=False, key="use_sector")
if use_sector:
    sector_weight_limit = st.sidebar.slider("行业权重上限 (%)", 10, 50, 20, key="sector_weight_limit")
    sector_weight = st.sidebar.slider("行业中性权重", 0.0, 1.0, 0.33, step=0.05, key="sector_weight")

# 市值中性
use_size = st.sidebar.checkbox("使用市值中性因子", value=False, key="use_size")
if use_size:
    size_window = st.sidebar.slider("市值代理计算窗口", 30, 120, 60, key="size_window")
    size_weight = st.sidebar.slider("市值中性权重", 0.0, 1.0, 0.33, step=0.05, key="size_weight")

# ------------------- 3. 风险预警与防御因子配置 -------------------
st.sidebar.divider()
st.sidebar.subheader("🛡️ 风险预警与防御因子")
st.sidebar.markdown("选择1-3种因子组合")

# 拥挤度因子
use_crowding = st.sidebar.checkbox("使用拥挤度因子", value=True, key="use_crowding")
if use_crowding:
    crowding_window = st.sidebar.slider("拥挤度计算窗口", 10, 60, 20, key="crowding_window")
    crowding_weight = st.sidebar.slider("拥挤度因子权重", 0.0, 1.0, 0.33, step=0.05, key="crowding_weight")

# 贝叶斯收缩
use_shrinkage = st.sidebar.checkbox("使用贝叶斯收缩因子", value=False, key="use_shrinkage")
if use_shrinkage:
    shrinkage_window = st.sidebar.slider("贝叶斯收缩窗口", 10, 60, 20, key="shrinkage_window")
    shrinkage_weight = st.sidebar.slider("贝叶斯收缩权重", 0.0, 1.0, 0.33, step=0.05, key="shrinkage_weight")

# 动态动能反转
use_dynamic_momentum = st.sidebar.checkbox("使用动态动能因子", value=False, key="use_dynamic_momentum")
if use_dynamic_momentum:
    dynamic_momentum_window = st.sidebar.slider("动态动能窗口", 10, 60, 20, key="dynamic_momentum_window")
    momentum_threshold = st.sidebar.slider("动量阈值", 1.0, 3.0, 2.0, step=0.1, key="momentum_threshold")
    dynamic_momentum_weight = st.sidebar.slider("动态动能权重", 0.0, 1.0, 0.33, step=0.05, key="dynamic_momentum_weight")



# ------------------- 回测和优化按钮 -------------------
st.sidebar.divider()
col4, col5 = st.sidebar.columns(2)
run_backtest = col4.button("运行回测")
optimize_weights = col5.button("一键优化")

# 风险厌恶系数设置
risk_aversion = st.sidebar.slider("风险厌恶系数", 0.1, 3.0, 1.0, step=0.1, help="值越大越保守，平衡风险和收益")

# ------------------- 监控功能 -------------------
st.sidebar.divider()
st.sidebar.subheader("🔍 策略监控")

# 监控名称输入
monitor_name = st.sidebar.text_input("监控名称", placeholder="输入监控策略名称")

# 邮箱订阅输入
email = st.sidebar.text_input("订阅邮箱", placeholder="your@email.com")

# 加入监控按钮
if st.sidebar.button("加入监控"):
    if monitor_name and email:
        # 生成唯一监控ID
        monitor_id = str(uuid.uuid4())[:8]
        
        # 构建监控信息
        monitor_info = {
            'id': monitor_id,
            'name': monitor_name,
            'ticker': ticker,
            'email': email,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'commission': commission,
            'risk_aversion': risk_aversion,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'active',
            'total_return': 0.0
        }
        
        # 保存到session_state
        st.session_state.monitor_list.append(monitor_info)
        
        st.success(f"监控 '{monitor_name}' 已创建！我们将向 {email} 发送实时交易信号")
        # 模拟实时信号生成和邮件发送
        st.info("监控系统已启动，将根据策略参数实时监控交易信号...")
    elif not monitor_name:
        st.error("请输入监控名称")
    else:
        st.error("请输入有效的邮箱地址")

# 显示监控列表
if st.session_state.monitor_list:
    st.sidebar.subheader("📋 监控列表")
    
    for monitor in st.session_state.monitor_list:
        with st.sidebar.expander(f"{monitor['name']} ({monitor['ticker']})", expanded=False):
            col1, col2 = st.sidebar.columns(2)
            col1.markdown(f"**ID**: {monitor['id']}")
            col2.markdown(f"**状态**: {monitor['status']}")
            st.sidebar.markdown(f"**邮箱**: {monitor['email']}")
            st.sidebar.markdown(f"**创建时间**: {monitor['created_at']}")
            st.sidebar.markdown(f"**累计收益**: {monitor['total_return']:.2f}%")
            
            # 操作按钮
            col3, col4 = st.sidebar.columns(2)
            if col3.button(f"删除", key=f"delete_{monitor['id']}"):
                st.session_state.monitor_list = [m for m in st.session_state.monitor_list if m['id'] != monitor['id']]
                st.experimental_rerun()
            
            if col4.button(f"查看收益", key=f"view_{monitor['id']}"):
                # 这里可以添加查看收益的逻辑，例如显示该监控的历史收益曲线
                st.session_state.current_monitor = monitor
                st.success(f"已加载监控 '{monitor['name']}' 的收益数据")

# ------------------- 回测执行 -------------------
if run_backtest or optimize_weights:
    with st.spinner("正在获取数据并计算..."):
        # 1. 检查因子选择数量
        macro_selected = sum([use_liquidity, use_volatility, use_credit_spread, use_inflation])
        neutral_selected = sum([use_beta, use_sector, use_size])
        risk_selected = sum([use_crowding, use_shrinkage, use_dynamic_momentum])
        
        if macro_selected < 1 or macro_selected > 3:
            st.error("宏观风险因子必须选择1-3种")
        elif neutral_selected < 1 or neutral_selected > 3:
            st.error("中性因子必须选择1-3种")
        elif risk_selected < 1 or risk_selected > 3:
            st.error("风险预警与防御因子必须选择1-3种")
        else:
            # 2. 获取数据
            fetcher = StockDataFetcher()
            df = fetcher.get_us_stock_daily(ticker)
            
            if df is not None:
                # 过滤时间
                df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
                
                # 3. 计算选中的因子
                fe = FactorEngine()
                
                # 计算宏观风险因子
                macro_factors = []
                if use_liquidity:
                    df = fe.add_liquidity_factor(df, liquidity_window)
                    macro_factors.append(('liquidity_factor', liquidity_weight))
                if use_volatility:
                    df = fe.add_volatility_term_structure(df, short_vol_window, long_vol_window)
                    macro_factors.append(('volatility_term_structure', volatility_weight))
                if use_credit_spread:
                    df = fe.add_credit_spread_proxy(df, credit_spread_window)
                    macro_factors.append(('credit_spread_proxy', credit_spread_weight))
                if use_inflation:
                    df = fe.add_inflation_rate_proxy(df, inflation_window)
                    macro_factors.append(('inflation_proxy', inflation_weight))
                
                # 计算中性因子
                neutral_factors = []
                if use_beta:
                    df = fe.add_beta_neutral(df, window=beta_window)
                    neutral_factors.append(('beta', beta_weight))
                if use_sector:
                    df = fe.add_sector_neutrality(df, sector_weight_limit / 100)
                    neutral_factors.append(('sector_neutral_signal', sector_weight))
                if use_size:
                    df = fe.add_size_neutrality(df, window=size_window)
                    neutral_factors.append(('size_neutral_signal', size_weight))
                
                # 计算风险预警与防御因子
                risk_factors = []
                if use_crowding:
                    df = fe.add_crowding_factor(df, window=crowding_window)
                    risk_factors.append(('crowding_factor', crowding_weight))
                if use_shrinkage:
                    df = fe.add_bayesian_shrinkage(df, window=shrinkage_window)
                    risk_factors.append(('bayesian_volatility', shrinkage_weight))
                if use_dynamic_momentum:
                    df = fe.add_dynamic_momentum_reversal(df, window=dynamic_momentum_window, threshold=momentum_threshold)
                    risk_factors.append(('dynamic_momentum_weight', dynamic_momentum_weight))
                
                # 4. 数据长度检查
                # 找出所有使用的窗口大小
                all_windows = []
                if use_volatility:
                    all_windows.extend([short_vol_window, long_vol_window])
                if use_inflation:
                    all_windows.append(inflation_window)
                if use_beta:
                    all_windows.append(beta_window)
                if use_size:
                    all_windows.append(size_window)
                
                if len(df) < max(all_windows, default=0):
                    st.error("数据长度不足以计算所选周期的因子，请增加时间范围。")
                else:
                    # 5. 定义因子权重
                    factor_weights = {}
                    # 合并所有选中的因子和权重
                    for factor_name, weight in macro_factors + neutral_factors + risk_factors:
                        factor_weights[factor_name] = weight
                    
                    # 6. 如果是优化权重，调用优化函数
                    if optimize_weights:
                        # 获取所有选中的因子名称
                        factor_list = [factor_name for factor_name, _ in macro_factors + neutral_factors + risk_factors]
                        
                        # 创建回测引擎
                        bt = VectorizedBacktester(initial_capital=initial_capital, commission=commission)
                        
                        # 优化权重
                        optimized_weights, best_score = bt.optimize_weights(df, factor_list, risk_aversion=risk_aversion)
                        
                        # 显示优化结果
                        st.success(f"因子权重优化完成！优化后的夏普比率: {best_score:.4f}")
                        
                        # 显示优化后的权重
                        st.subheader("🎯 优化后的因子权重")
                        weight_df = pd.DataFrame.from_dict(optimized_weights, orient='index', columns=['优化权重'])
                        weight_df['优化权重'] = weight_df['优化权重'].apply(lambda x: f"{x:.4f}")
                        st.table(weight_df)
                        
                        # 使用优化后的权重进行回测
                        factor_weights = optimized_weights
                    
                    # 7. 运行回测
                    bt = VectorizedBacktester(initial_capital=initial_capital, commission=commission)
                    results, metrics = bt.run(df, factor_weights=factor_weights)
                    
                    # 生成交易记录
                    trade_records = bt.get_trade_records(results, ticker)
                    
                    # --- 展示结果 ---
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("累计收益率", f"{metrics['Total Return (%)']:.2f}%")
                    col2.metric("年化收益率", f"{metrics['Annual Return (%)']:.2f}%")
                    col3.metric("夏普比率", f"{metrics['Sharpe Ratio']:.2f}")
                    col4.metric("最大回撤", f"{metrics['Max Drawdown (%)']:.2f}%")
                    col5.metric("最终资产", f"${results['equity'].iloc[-1]:,.2f}")
                    
                    # 交易记录表格
                    st.subheader("📋 交易记录")
                    if trade_records:
                        st.dataframe(trade_records, width='stretch')
                        st.metric("总交易次数", f"{len(trade_records)}")
                    else:
                        st.info("没有产生交易记录")
                    
                    # 绘制净值曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results.index, y=results['equity'], mode='lines', name='策略净值'))
                    fig.add_trace(go.Scatter(x=results.index, y=results['Close'] * (initial_capital / results['Close'].iloc[0]), 
                                           mode='lines', name='基准 (买入持有)', line=dict(dash='dash', color='gray')))
                    
                    fig.update_layout(title=f"{ticker} 策略表现 vs 基准", xaxis_title="日期", yaxis_title="资金净值 ($)", height=500)
                    st.plotly_chart(fig, width='stretch')
                    
                    # 数据表格
                    with st.expander("查看详细回测数据"):
                        # 选择关键列显示（只显示选中的因子相关列）
                        key_columns = ['Close', 'signal', 'buy_price', 'sell_price', 'equity']
                        # 添加选中的因子列
                        for factor_name, _ in macro_factors + neutral_factors + risk_factors:
                            key_columns.append(factor_name)
                        # 只显示存在的列
                        available_columns = [col for col in key_columns if col in results.columns]
                        st.dataframe(results[available_columns].tail(20), width='stretch')
                    
                    # 更新监控收益数据
                    if monitor_name:
                        # 查找对应的监控
                        for monitor in st.session_state.monitor_list:
                            if monitor['name'] == monitor_name and monitor['ticker'] == ticker:
                                # 更新收益数据
                                monitor['total_return'] = metrics['Total Return (%)']
                                monitor['status'] = 'active'
                                break
                    
                    # 监控信号展示
                    st.subheader("📡 实时监控信号")
                    # 模拟生成监控信号
                    latest_signal = results['signal'].iloc[-1]
                    latest_close = results['Close'].iloc[-1]
                    
                    if latest_signal == 1:
                        st.success(f"📈 买入信号！\n\n股票代码: {ticker}\n信号时间: {results.index[-1].strftime('%Y-%m-%d')}\n买入价格: ${latest_close:.2f}\n\n策略: {monitor_name if monitor_name else '未命名'}\n状态: 监控中")
                        # 模拟邮件发送
                        if email:
                            st.info(f"📧 交易信号已发送至: {email}")
                    elif latest_signal == 0:
                        st.info(f"📊 观望信号！\n\n股票代码: {ticker}\n信号时间: {results.index[-1].strftime('%Y-%m-%d')}\n当前价格: ${latest_close:.2f}\n\n策略: {monitor_name if monitor_name else '未命名'}\n状态: 监控中")
                    else:
                        st.warning(f"⚠️ 卖出信号！\n\n股票代码: {ticker}\n信号时间: {results.index[-1].strftime('%Y-%m-%d')}\n卖出价格: ${latest_close:.2f}\n\n策略: {monitor_name if monitor_name else '未命名'}\n状态: 监控中")
                        # 模拟邮件发送
                        if email:
                            st.info(f"📧 交易信号已发送至: {email}")
                    
                    # 显示当前监控的收益曲线
                    if 'current_monitor' in st.session_state:
                        st.subheader(f"📊 {st.session_state.current_monitor['name']} 收益曲线")
                        # 模拟收益曲线
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=results.index, y=results['equity'], mode='lines', name='策略净值'))
                        fig.add_trace(go.Scatter(x=results.index, y=results['Close'] * (initial_capital / results['Close'].iloc[0]), 
                                               mode='lines', name='基准 (买入持有)', line=dict(dash='dash', color='gray')))
                        fig.update_layout(title=f"监控 '{st.session_state.current_monitor['name']}' 收益表现", 
                                         xaxis_title="日期", yaxis_title="资金净值 ($)", height=400)
                        st.plotly_chart(fig, width='stretch')
            else:
                st.error(f"无法获取代码为 {ticker} 的股票数据，请检查输入或网络。")
else:
    st.info("请在左侧配置参数后点击 '开始运行回测' 或 '一键优化'")
