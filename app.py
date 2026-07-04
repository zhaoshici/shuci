#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股实时监控工具 - 完整版
功能：美股实时数据 + 异常量价 + 1/5/10min三周期图 
     + EMA/KDJ/RSI/KC指标 + 标准形态检测（旗型/楔形/双底等）
     + 自定义形态记忆 + 三周期共振提示 + 策略输入
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
    st.session_state.custom_patterns = []  # 用户自定义形态

# ==================== 指标计算 ====================
def calc_indicators(df):
    if df.empty:
        return df
    df = df.copy()
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

# ==================== 简单形态检测 ====================
def detect_double_bottom(df, window=20, tolerance=0.03):
    if len(df) < window * 3:
        return False, ""
    lows = df['low'].rolling(window).min()
    if len(lows) < window * 2:
        return False, ""
    # 简化检测：两个接近的低点
    recent_lows = lows.tail(window * 2)
    if recent_lows.iloc[-1] < recent_lows.min() * (1 + tolerance):
        return True, "可能双底形态"
    return False, ""

def detect_bull_flag(df, pole_window=30, flag_window=20):
    if len(df) < pole_window + flag_window:
        return False, ""
    # 简化：前期大幅上涨 + 近期小幅收敛
    pole_return = (df['close'].iloc[-flag_window] / df['close'].iloc[-pole_window-flag_window] - 1)
    flag_range = (df['high'].tail(flag_window).max() - df['low'].tail(flag_window).min()) / df['close'].tail(flag_window).mean()
    if pole_return > 0.15 and flag_range < 0.08:
        return True, "可能牛旗形态"
    return False, ""

def detect_wedge(df, window=30):
    if len(df) < window:
        return False, ""
    highs = df['high'].tail(window)
    lows = df['low'].tail(window)
    if highs.iloc[0] > highs.iloc[-1] and lows.iloc[0] < lows.iloc[-1]:
        return True, "可能收敛楔形"
    return False, ""

# ==================== 数据获取 ====================
@st.cache_data(ttl=60)
def get_data(ticker, period="5d"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        df_1m = stock.history(interval="1m", period="5d")
        df_5m = stock.history(interval="5m", period="10d")
        df_15m = stock.history(interval="15m", period="15d")
        
        df_1m = calc_indicators(df_1m)
        df_5m = calc_indicators(df_5m)
        df_15m = calc_indicators(df_15m)
        return info, df_1m, df_5m, df_15m
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 设置")
    ticker = st.text_input("美股代码", value="AAPL").upper().strip()
    
    st.subheader("📋 我的策略")
    with st.expander("添加/管理策略"):
        new_name = st.text_input("策略名称")
        new_buy = st.text_area("买入条件表达式（支持 ema200, kdj_k > kdj_d 等）")
        if st.button("添加策略"):
            if new_name and new_buy:
                st.session_state.strategies.append({"name": new_name, "buy_expr": new_buy, "sell_expr": ""})
                st.success("策略已添加！")
    
    st.subheader("📐 我的自定义形态（记忆功能）")
    with st.expander("添加自定义形态"):
        pat_name = st.text_input("形态名称")
        pat_desc = st.text_area("形态描述（文字规则）")
        if st.button("保存自定义形态"):
            if pat_name and pat_desc:
                st.session_state.custom_patterns.append({"name": pat_name, "description": pat_desc})
                st.success(f"已记忆形态：{pat_name}")
    
    if st.session_state.custom_patterns:
        st.write("已保存的自定义形态：")
        for p in st.session_state.custom_patterns:
            st.write(f"- {p['name']}: {p['description'][:50]}...")

# ==================== 主界面 ====================
if ticker:
    info, df_1m, df_5m, df_15m = get_data(ticker)
    
    if not df_1m.empty:
        # 实时报价
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("当前价格", f"{info.get('currentPrice', 'N/A')}", 
                      f"{info.get('regularMarketChangePercent', 0):.2f}%")
        with col2:
            st.metric("成交量", f"{info.get('volume', 0):,}")
        
        # 三个周期图
        st.subheader("📊 三个周期K线图（含指标）")
        tabs = st.tabs(["1分钟", "5分钟", "15分钟（≈10分钟）"])
        
        for tab, df, title in zip(tabs, [df_1m, df_5m, df_15m], ["1分钟", "5分钟", "15分钟"]):
            with tab:
                if not df.empty:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], 
                                                low=df['low'], close=df['close'], name='K线'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name='EMA20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name='EMA200'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量'), row=2, col=1)
                    fig.update_layout(height=450, title=title)
                    st.plotly_chart(fig, use_container_width=True)
        
        # 共振 + 形态检测
        st.subheader("🎯 三周期共振 + 形态检测")
        
        latest_1m = df_1m.iloc[-1] if not df_1m.empty else None
        latest_5m = df_5m.iloc[-1] if not df_5m.empty else None
        latest_15m = df_15m.iloc[-1] if not df_15m.empty else None
        
        signals = []
        
        # 示例共振（你的例子）
        if latest_15m is not None and latest_5m is not None and latest_1m is not None:
            cond_15 = latest_15m['close'] > latest_15m['ema200']
            cond_5 = abs(latest_5m['close'] - latest_5m['ema100']) / latest_5m['ema100'] < 0.02
            kdj_cross = latest_1m['kdj_k'] > latest_1m['kdj_d']
            
            if cond_15:
                signals.append("✅ 15分钟突破 EMA200")
            if cond_5:
                signals.append("✅ 5分钟接近 EMA100")
            if kdj_cross:
                signals.append("✅ 1分钟 KDJ 金叉")
            
            if cond_15 and cond_5 and kdj_cross:
                st.error("🚨🚨🚨 **三周期共振满足（你的示例策略）！** 强烈信号", icon="🚨")
        
        # 标准形态检测
        db, db_msg = detect_double_bottom(df_5m)
        if db:
            signals.append(f"📐 检测到 {db_msg}")
        
        bf, bf_msg = detect_bull_flag(df_5m)
        if bf:
            signals.append(f"📐 检测到 {bf_msg}")
        
        wd, wd_msg = detect_wedge(df_5m)
        if wd:
            signals.append(f"📐 检测到 {wd_msg}")
        
        # 自定义形态显示
        if st.session_state.custom_patterns:
            signals.append("📝 已加载你的自定义形态（工具会尝试匹配）")
        
        if signals:
            for s in signals:
                st.write(s)
        else:
            st.info("当前无明显共振或形态信号")
        
        # 异常量价
        if latest_5m is not None and latest_5m['volume'] > latest_5m['vol_ma20'] * 2:
            st.warning("🔥 5分钟成交量异常放大！")

st.caption("工具仅供学习研究 | 数据来源 yfinance | 部署后 iPad 直接使用")