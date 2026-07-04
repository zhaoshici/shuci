#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股实时监控工具 - 云部署优化版
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="美股实时监控 + 形态共振工具", layout="wide", page_icon="📈")
st.title("📈 美股实时异常监控 + 三周期共振 + 形态检测工具")
st.caption("支持美股 | 1/5/10分钟三周期 | EMA/KDJ/RSI/KC | 标准形态 + 自定义形态记忆 | 共振高亮提示")

# ==================== Session State ====================
if 'strategies' not in st.session_state:
    st.session_state.strategies = [
        {"name": "放量突破EMA200", "buy_expr": "close > ema200 and volume > vol_ma20 * 2", "sell_expr": "close < ema20"}
    ]
if 'custom_patterns' not in st.session_state:
    st.session_state.custom_patterns = []

# ==================== 指标计算 ====================
def calc_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    
    # 确保列名小写
    df.columns = [col.lower() for col in df.columns]
    
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['kdj_k'] = rsv.ewm(span=3).mean()
    df['kdj_d'] = df['kdj_k'].ewm(span=3).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # Keltner Channel
    df['tr'] = pd.concat([df['high']-df['low'], 
                          (df['high']-df['close'].shift()).abs(), 
                          (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = df['tr'].ewm(span=14).mean()
    df['kc_middle'] = df['close'].ewm(span=20).mean()
    df['kc_upper'] = df['kc_middle'] + 2 * df['atr']
    df['kc_lower'] = df['kc_middle'] - 2 * df['atr']
    
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    return df

# ==================== 形态检测（简化版） ====================
def detect_double_bottom(df, window=20, tolerance=0.03):
    if len(df) < window * 3:
        return False, ""
    try:
        lows = df['low'].rolling(window).min()
        recent_lows = lows.tail(window * 2)
        if recent_lows.iloc[-1] < recent_lows.min() * (1 + tolerance):
            return True, "可能双底形态"
    except:
        pass
    return False, ""

# ==================== 数据获取（云端优化版） ====================
@st.cache_data(ttl=60)
def get_data(ticker, period="5d"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        df_1m = stock.history(interval="1m", period="5d", prepost=False)
        df_5m = stock.history(interval="5m", period="10d", prepost=False)
        df_15m = stock.history(interval="15m", period="15d", prepost=False)
        
        df_1m = calc_indicators(df_1m)
        df_5m = calc_indicators(df_5m)
        df_15m = calc_indicators(df_15m)
        
        return info, df_1m, df_5m, df_15m
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 设置")
    ticker = st.text_input("美股代码", value="AAPL").upper().strip()
    
    st.subheader("📋 我的策略")
    with st.expander("添加/管理策略"):
        new_name = st.text_input("策略名称")
        new_buy = st.text_area("买入条件表达式")
        if st.button("添加策略"):
            if new_name and new_buy:
                st.session_state.strategies.append({"name": new_name, "buy_expr": new_buy, "sell_expr": ""})
                st.success("策略已添加！")

# ==================== 主界面 ====================
if ticker:
    info, df_1m, df_5m, df_15m = get_data(ticker)
    
    if not df_1m.empty and 'close' in df_1m.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            change = info.get('regularMarketChangePercent', 0)
            st.metric("当前价格", f"{current_price}", f"{change:.2f}%")
        with col2:
            st.metric("成交量", f"{info.get('volume', 0):,}")
        
        st.subheader("📊 三个周期K线图")
        tabs = st.tabs(["1分钟", "5分钟", "15分钟"])
        
        for tab, df, title in zip(tabs, [df_1m, df_5m, df_15m], ["1分钟", "5分钟", "15分钟"]):
            with tab:
                if not df.empty:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], 
                                                low=df['low'], close=df['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name='EMA20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df.get('ema200', None), name='EMA200'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量'), row=2, col=1)
                    fig.update_layout(height=450, title=title)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ 数据加载成功！")
    else:
        st.warning("正在加载数据或暂无数据，请稍等或换个股票代码试试（如 AAPL, TSLA, NVDA）")

st.caption("工具仅供学习研究 | 数据来源 yfinance | 部署后 iPad 直接使用")
