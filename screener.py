import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")


@st.cache_data
def load_data():
    """加载数据"""
    try:
        # 读取股票基本信息
        stock_basic = pd.read_csv('data/stock_basic.csv')

        # 读取并合并所有月度数据
        data_dir = 'data/data_by_month'
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

        if not all_files:
            st.error("未找到月度数据文件")
            return None, None

        # 使用列表存储每个月的数据
        monthly_data = []

        # 读取每个月的数据
        for file in all_files:
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                monthly_data.append(df)
            except Exception as e:
                st.warning(f"读取文件 {file} 时出错: {str(e)}")
                continue

        # 合并所有月度数据
        if not monthly_data:
            st.error("没有成功读取任何数据")
            return None, None

        daily_data = pd.concat(monthly_data, ignore_index=True)

        # 合并股票基本信息到日线数据中
        daily_data = pd.merge(
            daily_data,
            stock_basic[['ts_code', 'name']],
            on='ts_code',
            how='left'
        )

        # 将日期列转换为datetime
        daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'].astype(str))

        # 按日期排序
        daily_data = daily_data.sort_values('trade_date')

        return daily_data, stock_basic

    except Exception as e:
        st.error(f"加载数据时发生错误: {str(e)}")
        return None, None


class StockScreener:
    def __init__(self):
        try:
            # 使用缓存加载数据
            self.daily_data, self.stock_basic = load_data()

            # 验证必要的列是否存在
            required_daily_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close']
            required_basic_columns = ['ts_code', 'name']

            for col in required_daily_columns:
                if col not in self.daily_data.columns:
                    raise ValueError(f"日线数据缺少必要的列: {col}")

            for col in required_basic_columns:
                if col not in self.stock_basic.columns:
                    raise ValueError(f"基础信息数据缺少必要的列: {col}")

        except Exception as e:
            st.error(f"初始化时发生错误: {str(e)}")
            raise

    def get_latest_date(self):
        """获取最新交易日期"""
        return self.daily_data['trade_date'].max()

    @staticmethod
    @st.cache_data
    def filter_price_change(_daily_data, date1, date2, price_type1, price_type2, compare):
        """日期价格比较筛选"""
        try:
            # 获取两个日期的价格数据
            prices1 = _daily_data[_daily_data['trade_date'] == date1][['ts_code', price_type1, 'name']]
            prices2 = _daily_data[_daily_data['trade_date'] == date2][['ts_code', price_type2]]

            # 重命名列，避免合并后的命名冲突
            prices1 = prices1.rename(columns={price_type1: 'price1'})
            prices2 = prices2.rename(columns={price_type2: 'price2'})

            if prices1.empty or prices2.empty:
                st.warning(f"未找到指定日期的数据")
                return pd.DataFrame()

            # 合并数据
            merged = pd.merge(prices1, prices2, on='ts_code')

            # 根据条件筛选
            if compare == '>':
                filtered = merged[merged['price1'] > merged['price2']]
            elif compare == '>=':
                filtered = merged[merged['price1'] >= merged['price2']]
            elif compare == '<':
                filtered = merged[merged['price1'] < merged['price2']]
            elif compare == '<=':
                filtered = merged[merged['price1'] <= merged['price2']]

            return filtered

        except Exception as e:
            st.error(f"日期价格比较筛选时发生错误: {str(e)}")
            return pd.DataFrame()

    def get_price_change_stocks(self, date1, date2, price_type1, price_type2, compare):
        """日期价格比较筛选的包装方法"""
        return self.filter_price_change(self.daily_data, date1, date2, price_type1, price_type2, compare)

    @staticmethod
    @st.cache_data
    def filter_price_range(_daily_data, date, price_type, min_price, max_price):
        """价格区间筛选"""
        try:
            # 获取指定日期的价格数据
            stocks = _daily_data[_daily_data['trade_date'] == date][['ts_code', price_type, 'name']]

            if stocks.empty:
                st.warning(f"未找到指定日期的数据")
                return pd.DataFrame()

            # 根据价格区间筛选
            filtered = stocks[
                (stocks[price_type] >= min_price) &
                (stocks[price_type] <= max_price)
                ]

            return filtered

        except Exception as e:
            st.error(f"价格区间筛选时发生错误: {str(e)}")
            return pd.DataFrame()

    def get_price_range_stocks(self, date, price_type, min_price, max_price):
        """价格区间筛选的包装方法"""
        return self.filter_price_range(self.daily_data, date, price_type, min_price, max_price)

    @staticmethod
    @st.cache_data
    def filter_stock_code(_daily_data, _latest_date, code_pattern):
        """股票代码筛选"""
        try:
            # 获取最新的交易日数据
            stocks = _daily_data[_daily_data['trade_date'] == _latest_date][['ts_code', 'close', 'name']]

            # 使用模糊匹配筛选股票代码
            filtered = stocks[
                stocks['ts_code'].str.contains(code_pattern, case=False, na=False) |
                stocks['name'].str.contains(code_pattern, case=False, na=False)
                ]

            return filtered

        except Exception as e:
            st.error(f"股票代码筛选时发生错误: {str(e)}")
            return pd.DataFrame()

    def get_stock_code_filter(self, code_pattern):
        """股票代码筛选的包装方法"""
        latest_date = self.get_latest_date()
        return self.filter_stock_code(self.daily_data, latest_date, code_pattern)

    @staticmethod
    @st.cache_data
    def filter_consecutive_trend(_daily_data, days, direction='up', min_pct=0):
        """连续涨跌筛选（支持上涨或下跌）"""
        try:
            # 获取所有交易日期并排序
            trade_dates = sorted(_daily_data['trade_date'].unique(), reverse=True)
            if not trade_dates:
                return pd.DataFrame()

            # 只处理最近60个交易日的数据来提高性能
            recent_dates = trade_dates[:60]
            df = _daily_data[_daily_data['trade_date'].isin(recent_dates)].copy()

            # 按股票和日期排序
            df = df.sort_values(['ts_code', 'trade_date'], ascending=[True, False])

            # 计算涨跌幅
            df['pct_change'] = df.groupby('ts_code')['close'].pct_change(-1) * 100

            def check_consecutive_trend(group):
                """检查连续涨跌趋势"""
                if len(group) < days:
                    return False
                # 获取排序后的交易日
                dates = sorted(group['trade_date'].unique(), reverse=True)

                # 检查最近days个交易日的涨跌幅
                recent_records = group[group['trade_date'].isin(dates[:days])]
                if len(recent_records) != days:  # 确保有足够的连续交易日数据
                    return False

                # 根据方向检查涨跌
                if direction == 'up':
                    return all(recent_records['pct_change'] > min_pct)
                else:  # direction == 'down'
                    return all(recent_records['pct_change'] < -min_pct)

            # 找出符合条件的股票
            valid_stocks = []
            for ts_code, group in df.groupby('ts_code'):
                if check_consecutive_trend(group):
                    # 获取最新一天的数据
                    latest = group.iloc[0]
                    # 计算总涨跌幅
                    total_change = group.head(days)['pct_change'].sum()
                    valid_stocks.append({
                        'ts_code': ts_code,
                        'name': latest['name'],
                        'latest_price': latest['close'],
                        'consecutive_days': days,
                        'total_change': total_change,
                        'latest_date': latest['trade_date']
                    })

            if not valid_stocks:
                return pd.DataFrame()

            result_df = pd.DataFrame(valid_stocks)
            return result_df

        except Exception as e:
            st.error(f"连续涨跌筛选时发生错误: {str(e)}")
            return pd.DataFrame()

    def get_consecutive_trend_stocks(self, days, direction='up', min_pct=0):
        """连续涨跌筛选的包装方法"""
        return self.filter_consecutive_trend(self.daily_data, days, direction, min_pct)

    @staticmethod
    @st.cache_data
    def get_stock_data_static(_daily_data, ts_code):
        """获取单个股票的所有数据"""
        try:
            stock_data = _daily_data[_daily_data['ts_code'] == ts_code].copy()
            stock_data = stock_data.sort_values('trade_date')
            stock_data = stock_data[
                (stock_data['open'] > 0) &
                (stock_data['high'] > 0) &
                (stock_data['low'] > 0) &
                (stock_data['close'] > 0) &
                stock_data['open'].notna() &
                stock_data['high'].notna() &
                stock_data['low'].notna() &
                stock_data['close'].notna()
                ]
            return stock_data
        except Exception as e:
            st.error(f"获取股票数据时发生错误: {str(e)}")
            return pd.DataFrame()

    def get_stock_data(self, ts_code):
        """获取单个股票数据的包装方法"""
        return self.get_stock_data_static(self.daily_data, ts_code)

    @staticmethod
    def create_candlestick(_stock_data):
        """创建K线图"""
        try:
            if _stock_data.empty:
                st.warning("没有找到股票数据")
                return None

            stock_name = _stock_data['name'].iloc[0]
            ts_code = _stock_data['ts_code'].iloc[0]

            # 计算涨跌幅
            _stock_data['pct_change'] = (_stock_data['close'] - _stock_data['close'].shift(1)) / _stock_data['close'].shift(1) * 100

            fig = go.Figure(data=[go.Candlestick(
                x=_stock_data['trade_date'],
                open=_stock_data['open'],
                high=_stock_data['high'],
                low=_stock_data['low'],
                close=_stock_data['close'],
                increasing_line_color='#ff7f7f',
                decreasing_line_color='#7fbf7f',
                hovertext=[
                    f'日期: {date:%Y-%m-%d}<br>' +
                    f'开盘: {open:.2f}<br>' +
                    f'最高: {high:.2f}<br>' +
                    f'最低: {low:.2f}<br>' +
                    f'收盘: {close:.2f}<br>' +
                    f'涨跌幅: {pct_change:+.2f}%'
                    for date, open, high, low, close, pct_change in zip(
                        _stock_data['trade_date'],
                        _stock_data['open'],
                        _stock_data['high'],
                        _stock_data['low'],
                        _stock_data['close'],
                        _stock_data['pct_change']
                    )
                ]
            )])

            # 计算默认显示的时间范围(最近6个月)
            end_date = _stock_data['trade_date'].max()
            start_date = end_date - pd.DateOffset(months=6)

            fig.update_layout(
                title=f'{stock_name}({ts_code})',
                autosize=True,
                width=1700,
                height=800,
                yaxis={
                    'autorange': True,
                    'fixedrange': False
                },
                xaxis={
                    'rangeslider': {'visible': False},
                    'type': 'date',
                    'range': [start_date, end_date],
                    'fixedrange': False
                },
                dragmode='pan',
                showlegend=False,
            )

            # 添加时间范围选择按钮
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1月", step="month", stepmode="backward"),
                        dict(count=6, label="6月", step="month", stepmode="backward"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all", label="全部")
                    ])
                )
            )

            # 更新 Y 轴设置
            fig.update_yaxes(
                tickformat='.2f',
                gridcolor='lightgrey',
                autorange=True,
                fixedrange=False
            )

            return fig

        except Exception as e:
            st.error(f"绘制K线图时发生错误: {str(e)}")
            return None

    def plot_candlestick(self, ts_code):
        """绘制K线图的包装方法"""
        stock_data = self.get_stock_data(ts_code)
        return self.create_candlestick(stock_data)


def main():
    try:
        # 初始化
        screener = StockScreener()

        # 侧边栏:筛选条件
        st.sidebar.header('筛选条件')

        # 使用 session_state 来存储筛选条件
        if 'filter_conditions' not in st.session_state:
            st.session_state.filter_conditions = []

        # 筛选器类型
        filter_types = {
            'stock_code': '股票',
            'consecutive_trend': '连续涨跌',
            'date_price_compare': '价格比较',
            'price_range': '价格区间'
        }

        # 价格类型选项
        price_types = {
            'open': '开盘价',
            'close': '收盘价',
            'high': '最高价',
            'low': '最低价'
        }

        # 逻辑运算符
        logic_operators = {
            'AND': '与',
            'OR': '或',
            'NOT': '非'
        }

        # 添加筛选条件的按钮
        if st.sidebar.button('添加筛选条件'):
            st.session_state.filter_conditions.append({
                'type': 'date_price_compare',
                'logic': 'AND'
            })

        # 显示现有的筛选条件
        for i, condition in enumerate(st.session_state.filter_conditions):
            st.sidebar.markdown(f"### 筛选条件 {i + 1}")

            # 选择筛选器类型
            condition['type'] = st.sidebar.selectbox(
                "筛选类型",
                options=list(filter_types.keys()),
                format_func=lambda x: filter_types[x],
                key=f'type_{i}'
            )

            # 根据不同的筛选器类型显示不同的选项
            if condition['type'] == 'date_price_compare':
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    condition['date1'] = st.date_input("日期1", key=f'date1_{i}')
                    condition['price_type1'] = st.selectbox(
                        "价格类型1",
                        options=list(price_types.keys()),
                        format_func=lambda x: price_types[x],
                        key=f'price_type1_{i}'
                    )
                with col2:
                    condition['date2'] = st.date_input("日期2", key=f'date2_{i}')
                    condition['price_type2'] = st.selectbox(
                        "价格类型2",
                        options=list(price_types.keys()),
                        format_func=lambda x: price_types[x],
                        key=f'price_type2_{i}'
                    )
                condition['compare'] = st.sidebar.selectbox(
                    "比较条件",
                    options=['>', '<='],
                    format_func=lambda x: {'>': '大于', '<=': '小于等于'}[x],
                    key=f'compare_{i}'
                )

            elif condition['type'] == 'price_range':
                condition['date'] = st.sidebar.date_input("日期", key=f'date_{i}')
                condition['price_type'] = st.sidebar.selectbox(
                    "价格类型",
                    options=list(price_types.keys()),
                    format_func=lambda x: price_types[x],
                    key=f'price_type_{i}'
                )
                condition['min_price'], condition['max_price'] = st.sidebar.slider(
                    "价格区间",
                    0.0, 200.0, (0.0, 200.0),
                    key=f'price_range_{i}'
                )

            elif condition['type'] == 'stock_code':
                condition['code'] = st.sidebar.text_input(
                    "股票代码或名称",
                    key=f'code_{i}'
                )

            elif condition['type'] == 'consecutive_trend':
                condition['direction'] = st.sidebar.radio(
                    "涨跌方向",
                    options=['up', 'down'],
                    format_func=lambda x: "上涨" if x == 'up' else "下跌",
                    key=f'trend_direction_{i}'
                )
                condition['days'] = st.sidebar.slider(
                    "连续天数",
                    1, 10, 3,
                    key=f'consecutive_days_{i}'
                )
                condition['min_pct'] = st.sidebar.slider(
                    "最小幅度(%)",
                    0.0, 5.0, 0.0,
                    step=0.1,
                    key=f'min_pct_{i}'
                )

            # 如果不是第一个条件,显示逻辑运算符
            if i > 0:
                condition['logic'] = st.sidebar.selectbox(
                    "逻辑运算符",
                    options=list(logic_operators.keys()),
                    format_func=lambda x: logic_operators[x],
                    key=f'logic_{i}'
                )

            # 删除当前筛选条件的按钮
            if st.sidebar.button('删除此条件', key=f'delete_{i}'):
                st.session_state.filter_conditions.pop(i)

        # 执行筛选的按钮
        if st.sidebar.button('执行筛选'):
            results = None
            for i, condition in enumerate(st.session_state.filter_conditions):
                current_result = None

                if condition['type'] == 'date_price_compare':
                    current_result = screener.get_price_change_stocks(
                        pd.to_datetime(condition['date1']),
                        pd.to_datetime(condition['date2']),
                        condition['price_type1'],
                        condition['price_type2'],
                        condition['compare']
                    )
                elif condition['type'] == 'price_range':
                    current_result = screener.get_price_range_stocks(
                        pd.to_datetime(condition['date']),
                        condition['price_type'],
                        condition['min_price'],
                        condition['max_price']
                    )
                elif condition['type'] == 'stock_code':
                    current_result = screener.get_stock_code_filter(condition['code'])
                elif condition['type'] == 'consecutive_trend':
                    current_result = screener.get_consecutive_trend_stocks(
                        condition['days'],
                        condition['direction'],
                        condition['min_pct']
                        )

                # 组合结果
                if i == 0:
                    results = current_result
                else:
                    if condition['logic'] == 'AND':
                        results = pd.merge(results, current_result, on=['ts_code', 'name'])
                    elif condition['logic'] == 'OR':
                        results = pd.concat([results, current_result]).drop_duplicates()
                    elif condition['logic'] == 'NOT':
                        results = results[~results['ts_code'].isin(current_result['ts_code'])]

            # 保存筛选结果到 session_state
            if results is not None and not results.empty:
                st.session_state.filtered_results = results

        # 显示筛选结果（如果存在）
        if 'filtered_results' in st.session_state:
            results = st.session_state.filtered_results
            st.write(f'符合股票: {len(results)}')

            # 创建固定位置的K线图容器
            chart_container = st.container()

            # 添加CSS来固定K线图位置
            st.markdown("""
                <style>
                    .stPlotlyChart {
                        position: fixed;
                        top: 60px;
                        left: 700px;
                        z-index: 1000;
                    }
                    .stock-list {
                        width: 200px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # 创建股票列表容器
            with st.container():
                st.markdown('<div class="stock-list">', unsafe_allow_html=True)
                stock_options = [f"{row['name']} ({row['ts_code']})" for _, row in results.iterrows()]
                selected_stock = st.radio("股票列表", stock_options, label_visibility="hidden")
                st.markdown('</div>', unsafe_allow_html=True)

            # 在固定位置显示K线图
            with chart_container:
                if selected_stock:
                    # 从选项中提取股票代码
                    ts_code = selected_stock.split('(')[1].split(')')[0]
                    fig = screener.plot_candlestick(ts_code)
                    if fig is not None:
                        config = {
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtons': [
                                ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                                ['toImage']
                            ]
                        }
                        st.plotly_chart(fig, config=config, use_container_width=False)

    except Exception as e:
        st.error(f'程序运行时发生错误: {str(e)}')


if __name__ == "__main__":
    main()
