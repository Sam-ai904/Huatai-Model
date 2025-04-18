#!pip install tushare bottleneck scikit-learn
import tushare as ts
import pandas as pd
import numpy as np
import random
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import bottleneck as bn
import matplotlib.pyplot as plt

# 设置Tushare API Token并初始化
ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')
pro = ts.pro_api()

# 获取沪深300成分股前100
def get_hs300_top50(start_date, end_date):
    try:
        df = pro.index_weight(index_code='000300.SH', start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError("未获取到沪深300成分股数据")

        df = df.sort_values('weight', ascending=False)
        stock_list = df['con_code'].unique()[:100].tolist()
        print(f"成功获取沪深300前100只股票: {stock_list[:6]}...（共 {len(stock_list)} 只）")
        return stock_list
    except Exception as e:
        print(f"获取沪深300成分股失败: {e}")
        return []

# 获取股票日频数据
def get_data(start_date, end_date, stock_list):
    df_list = []
    for stock in stock_list:
        try:
            temp_df = pro.daily(
                ts_code=stock,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,close,high,low,vol,pct_chg'
            )
            if not temp_df.empty:
                df_list.append(temp_df)
            else:
                print(f"股票 {stock} 在 {start_date} 至 {end_date} 无数据")
        except Exception as e:
            print(f"获取股票 {stock} 数据失败: {e}")

    if not df_list:
        print(f"时间范围 {start_date} 至 {end_date} 无任何股票数据")
        return pd.DataFrame()

    try:
        df = pd.concat(df_list)
        df.rename(columns={'vol': 'volume', 'pct_chg': 'return'}, inplace=True)
        df['return'] = df['return'] / 100
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        duplicates = df.duplicated(subset=['trade_date', 'ts_code'], keep=False)
        if duplicates.any():
            print(f"警告: 发现重复数据，共 {duplicates.sum()} 条，自动保留最后一条")
            df = df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='last')

        df_pivot = df.pivot(index='trade_date', columns='ts_code')

        for col in df_pivot.columns.levels[0]:
            df_pivot[col] = df_pivot[col].fillna(method='ffill').fillna(method='bfill')

        if 'return' in df_pivot:
            return_stats = df_pivot['return'].describe()
            print(f"收益率统计: {return_stats}")
            valid_stocks = return_stats.loc['std'] > 0
            valid_stocks = valid_stocks[valid_stocks].index.tolist()
            if not valid_stocks:
                print("所有股票的收益率均为常数或 NaN，无法继续")
                return pd.DataFrame()
            df_pivot = df_pivot.loc[:, df_pivot.columns.get_level_values(1).isin(valid_stocks)]

        return df_pivot
    except Exception as e:
        print(f"数据合并失败: {e}")
        return pd.DataFrame()

# 中位数去极值函数
def winsorize_median(factor, n_mad=5):
    factor = factor.copy()
    median = np.nanmedian(factor)
    mad = np.nanmedian(np.abs(factor - median))
    upper = median + n_mad * mad
    lower = median - n_mad * mad
    factor = np.clip(factor, lower, upper)
    return factor

# 标准化函数
def standardize(factor):
    factor = factor.copy()
    mean = np.nanmean(factor)
    std = np.nanstd(factor)
    return (factor - mean) / (std + 1e-10)

# 计算所有因子的值
def calculate_all_factors(data, factor_expressions, target_shape):
    factors = []
    expected_length = target_shape[0] * target_shape[1] if len(target_shape) > 1 else target_shape[0]
    for expr in factor_expressions:
        try:
            factor_values = eval(expr, {'np': np, 'bn': bn}, {'data': data})
            if isinstance(factor_values, pd.DataFrame):
                factor_values_flat = factor_values.values.flatten()
            else:
                factor_values_flat = factor_values.flatten()
            if len(factor_values_flat) != expected_length:
                print(f"因子 {expr} 的展平长度 {len(factor_values_flat)} 与目标长度 {expected_length} 不一致，调整中...")
                if len(factor_values_flat) > expected_length:
                    factor_values_flat = factor_values_flat[:expected_length]
                else:
                    factor_values_flat = np.pad(factor_values_flat, (0, expected_length - len(factor_values_flat)),
                                               mode='constant', constant_values=np.nan)
            factor_values_flat = standardize(winsorize_median(factor_values_flat))
            factors.append(factor_values_flat)
        except Exception as e:
            print(f"计算因子 {expr} 失败: {e}")
            factors.append(np.full(expected_length, np.nan))

    factors = np.array(factors).T
    return factors

# 使用随机森林进行因子合成
def synthesize_with_random_forest(factors, target, test_size=0.2, random_state=42):
    if factors.shape[0] != len(target):
        raise ValueError(f"factors 形状 {factors.shape[0]} 与 target 长度 {len(target)} 不一致")

    valid_mask = ~np.any(np.isnan(factors), axis=1) & ~np.isnan(target)
    factors_clean = factors[valid_mask]
    target_clean = target[valid_mask]

    if len(factors_clean) < 2:
        print("有效数据点少于2，无法进行随机森林训练")
        return np.full(len(target), np.nan), None

    X_train, X_test, y_train, y_test = train_test_split(
        factors_clean, target_clean, test_size=test_size, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Train MSE: {mse_train:.6f}, Test MSE: {mse_test:.6f}")

    final_factor = np.full(len(target), np.nan)
    final_factor[valid_mask] = rf.predict(factors[valid_mask])
    final_factor = standardize(final_factor)

    return final_factor, rf

# 保存结果
def save_results(final_factor, data, output_path="synthetic_factor.csv"):
    if len(final_factor) != len(data.index) * len(data['return'].columns):
        print(f"final_factor 长度 {len(final_factor)} 与预期长度 {len(data.index) * len(data['return'].columns)} 不一致，调整中...")
        min_length = min(len(final_factor), len(data.index) * len(data['return'].columns))
        final_factor = final_factor[:min_length]
        n_stocks = len(data['return'].columns)
        n_dates = min_length // n_stocks
        data_subset = data.iloc[:n_dates]
    else:
        data_subset = data
        n_stocks = len(data['return'].columns)
        n_dates = len(data.index)

    factor_values = final_factor.reshape(n_dates, n_stocks)
    result_df = pd.DataFrame(factor_values, index=data_subset.index, columns=data_subset['return'].columns)
    result_df.to_csv(output_path)
    print(f"合成因子已保存至 {output_path}")

# 初始化种群
def initialize_population(size, function_list):
    population = []
    for _ in range(size):
        formula = random.choice(function_list)
        population.append(formula)
    return population

# 计算适应度（RankIC + 互信息）
def calculate_fitness(formula, data, rankic_weight=0.5, mi_weight=0.5):
    try:
        factor_values = eval(formula, {'np': np, 'bn': bn}, {'data': data})
        if isinstance(factor_values, pd.DataFrame):
            factor_values_flat = factor_values.values.flatten()
        else:
            factor_values_flat = factor_values.flatten()
        returns_flat = data['return'].values.flatten()

        factor_values_flat = np.clip(factor_values_flat, -10, 10)
        returns_flat = np.clip(returns_flat, -0.5, 0.5)

        print(f"因子值统计: min={np.nanmin(factor_values_flat):.4f}, max={np.nanmax(factor_values_flat):.4f}, std={np.nanstd(factor_values_flat):.4f}")
        print(f"收益率统计: min={np.nanmin(returns_flat):.4f}, max={np.nanmax(returns_flat):.4f}, std={np.nanstd(returns_flat):.4f}")

        mask = ~(np.isnan(factor_values_flat) | np.isnan(returns_flat))
        if mask.sum() < 2:
            print("有效数据点少于 2，无法计算相关性或互信息")
            return -1

        if np.nanstd(factor_values_flat[mask]) == 0 or np.nanstd(returns_flat[mask]) == 0:
            print("因子值或收益率是常数，无法计算相关性或互信息")
            return -1

        rankic, _ = spearmanr(factor_values_flat[mask], returns_flat[mask])
        rankic = rankic if not np.isnan(rankic) else -1
        print(f"RankIC: {rankic:.4f}")

        X = factor_values_flat[mask].reshape(-1, 1)
        y = returns_flat[mask]
        mi = mutual_info_regression(X, y, random_state=42)[0]
        mi = mi if not np.isnan(mi) else 0
        print(f"Mutual Information: {mi:.4f}")

        fitness = rankic_weight * abs(rankic) + mi_weight * mi
        print(f"综合适应度: {fitness:.4f}")
        return fitness
    except Exception as e:
        print(f"计算适应度失败: {e}")
        return -1

# 进化种群
def evolve_population(population, data, generations, rankic_weight=0.5, mi_weight=0.5):
    for gen in range(generations):
        fitness_scores = [(formula, calculate_fitness(formula, data, rankic_weight, mi_weight)) for formula in population]
        scores = [score for _, score in fitness_scores if not np.isnan(score)]
        print(f"第 {gen+1} 代，适应度统计: min={min(scores) if scores else 'N/A'}, max={max(scores) if scores else 'N/A'}, mean={np.mean(scores) if scores else 'N/A'}")

        fitness_scores = sorted(fitness_scores, key=lambda x: x[1], reverse=True)[:int(len(fitness_scores)*0.6)]
        population = [item[0] for item in fitness_scores if item[1] > 0.015]

        print(f"第 {gen+1} 代，筛选后种群大小: {len(population)}")

        if not population:
            print("种群为空，停止进化，可能是因子公式无效或数据问题")
            return []

        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(population, 2)
            new_formula = f"({parent1}) + ({parent2})"
            if random.random() < 0.1:
                operations = ['+', '*', '-']
                new_formula = new_formula.replace('+', random.choice(operations))
            new_population.append(new_formula)
        population = new_population[:len(population)]
    return population

# 计算残差收益率
def calculate_residual_return(data, factor_pool):
    if not factor_pool:
        residual = data['return'].values.flatten()
        print(f"初始残差收益率形状: {data['return'].shape}, 展平后: {residual.shape}")
        return pd.Series(residual)
    try:
        X = np.column_stack([eval(formula, {'np': np, 'bn': bn}, {'data': data}) for formula in factor_pool])
        y = data['return'].values.flatten()
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 2:
            print("残差计算数据点不足，返回原始收益率")
            return pd.Series(y)
        lr = LinearRegression()
        lr.fit(X[mask], y[mask])
        residuals = y - lr.predict(X)
        print(f"残差计算后形状: {residuals.shape}")
        return pd.Series(residuals)
    except Exception as e:
        print(f"残差计算错误: {e}")
        return pd.Series(data['return'].values.flatten())

# 因子挖掘
def rolling_factor_extraction(start_date, end_date, interval_years=2, rankic_weight=0.5, mi_weight=0.5):
    stock_list = get_hs300_top50(start_date, end_date)
    if not stock_list:
        print("无法获取股票列表，退出")
        return [], None

    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    factor_pool = []
    function_list = [
        "data['open'] / (data['close'] + 1e-10)",
        "(data['high'] - data['low']) / (data['close'] + 1e-10)",
        "bn.move_mean(data['volume'], window=5) / (bn.move_mean(data['volume'], window=20) + 1e-10)",
        "np.log(data['close'] + 1) - np.log(bn.move_mean(data['close'], window=5) + 1)",
        "bn.nanrankdata(data['close'].diff(1), axis=0) / (data['close'] + 1e-10)",
        "(data['close'] - bn.move_mean(data['close'], window=10)) / bn.move_std(data['close'], window=10)"
    ]

    earliest_date = pd.to_datetime('20150101')
    last_data = None

    while current_date < end_date:
        sample_start = (current_date - pd.DateOffset(years=2))
        if sample_start < earliest_date:
            sample_start = earliest_date
        sample_start = sample_start.strftime('%Y%m%d')
        sample_end = current_date.strftime('%Y%m%d')

        print(f"正在处理窗口: {sample_start} 至 {sample_end}")
        data = get_data(sample_start, sample_end, stock_list)

        if data.empty:
            print(f"无数据: {sample_start} 至 {sample_end}")
            current_date += pd.DateOffset(years=interval_years)
            continue

        last_data = data

        population_size = 200
        generations = 2
        population = initialize_population(population_size, function_list)

        try:
            residual_return = calculate_residual_return(data, factor_pool)
            if isinstance(residual_return, pd.Series):
                residual_return_flat = residual_return.values
            else:
                residual_return_flat = residual_return.flatten()
            print(f"残差收益率统计: min={np.nanmin(residual_return_flat):.4f}, max={np.nanmax(residual_return_flat):.4f}, std={np.nanstd(residual_return_flat):.4f}")

            print("开始进化种群...")
            final_population = evolve_population(population, data, generations, rankic_weight, mi_weight)
            print(f"最终种群大小: {len(final_population)}")

            for formula in final_population:
                try:
                    factor_values = eval(formula, {'np': np, 'bn': bn}, {'data': data})
                    if isinstance(factor_values, pd.DataFrame):
                        factor_values_flat = factor_values.values.flatten()
                    else:
                        factor_values_flat = factor_values.flatten()
                    mask = ~(np.isnan(factor_values_flat) | np.isnan(residual_return_flat))
                    if mask.sum() < 2:
                        print(f"因子 {formula} 有效数据点少于 2，跳过")
                        continue

                    rankic, _ = spearmanr(factor_values_flat[mask], residual_return_flat[mask])
                    rankic = rankic if not np.isnan(rankic) else -1

                    X = factor_values_flat[mask].reshape(-1, 1)
                    y = residual_return_flat[mask]
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    mi = mi if not np.isnan(mi) else 0

                    fitness = rankic_weight * abs(rankic) + mi_weight * mi
                    if fitness > 0.015 and abs(rankic) < 0.7 and not np.isnan(fitness):
                        factor_pool.append(formula)
                        print(f"添加因子: {formula}, RankIC: {rankic:.4f}, MI: {mi:.4f}, Fitness: {fitness:.4f}")
                except Exception as e:
                    print(f"因子 {formula} 处理失败: {e}")
                    continue
        except Exception as e:
            print(f"处理错误: {e}")
            continue

        current_date += pd.DateOffset(years=interval_years)
        factor_pool = factor_pool[-100:]

    return factor_pool, last_data

# 回测函数
def backtest_synthetic_factor(start_date, end_date, stock_list, factor_pool, rf_model):
    if not factor_pool or rf_model is None:
        print("因子池为空或随机森林模型未训练，无法进行回测")
        return

    data = get_data(start_date, end_date, stock_list)
    if data.empty:
        print(f"回测时间范围 {start_date} 至 {end_date} 无数据")
        return

    dates = data.index
    stocks = data['return'].columns
    n_dates = len(dates)
    n_stocks = len(stocks)
    target_shape = (n_dates, n_stocks)

    factors = calculate_all_factors(data, factor_pool, target_shape)
    target = data['return'].shift(-1).values.flatten()

    if factors.shape[0] != len(target):
        print(f"factors 形状 {factors.shape[0]} 与 target 长度 {len(target)} 不一致，调整中...")
        min_length = min(factors.shape[0], len(target))
        factors = factors[:min_length]
        target = target[:min_length]

    final_factor, _ = synthesize_with_random_forest(factors, target)
    factor_values = final_factor.reshape(n_dates, n_stocks)
    factor_df = pd.DataFrame(factor_values, index=dates, columns=stocks)

    returns = data['return']

    portfolio_returns = []
    for date in returns.index:
        if date not in factor_df.index:
            portfolio_returns.append(0.0)
            continue
        factor_values_day = factor_df.loc[date]
        returns_day = returns.loc[date]

        valid_mask = ~factor_values_day.isna() & ~returns_day.isna()
        factor_values_day = factor_values_day[valid_mask]
        returns_day = returns_day[valid_mask]

        if len(factor_values_day) < 2:
            portfolio_returns.append(0.0)
            continue

        factor_rank = factor_values_day.rank()
        total_stocks = len(factor_rank)
        top_threshold = int(total_stocks * 0.8)
        bottom_threshold = int(total_stocks * 0.2)

        long_stocks = factor_rank[factor_rank > top_threshold].index
        short_stocks = factor_rank[factor_rank <= bottom_threshold].index

        long_return = returns_day[long_stocks].mean() if len(long_stocks) > 0 else 0.0
        short_return = returns_day[short_stocks].mean() if len(short_stocks) > 0 else 0.0

        portfolio_return = long_return - short_return
        portfolio_returns.append(portfolio_return if not np.isnan(portfolio_return) else 0.0)

    portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    n_days = len(portfolio_returns)
    n_years = n_days / 252
    annualized_return = (cumulative_returns.iloc[-1]) ** (1 / n_years) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    max_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max()
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0.0

    print("\n回测结果：")
    print(f"累计收益率：{cumulative_returns.iloc[-1] - 1:.4f}")
    print(f"年化收益率：{annualized_return:.4f}")
    print(f"年化波动率：{annualized_volatility:.4f}")
    print(f"最大回撤：{max_drawdown:.4f}")
    print(f"夏普比率：{sharpe_ratio:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='累计收益率')
    plt.title('合成因子多空组合累计收益率')
    plt.xlabel('日期')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid()
    plt.savefig('backtest_cumulative_return.png')
    plt.close()

# 主函数
def main():
    start_date = '20180101'
    end_date = '20240101'
    backtest_start_date = '20240101'
    backtest_end_date = '20241231'  
    print(f"因子挖掘范围: {start_date} 至 {end_date}")

    # 因子挖掘
    rolling_factors, data = rolling_factor_extraction(start_date, end_date, rankic_weight=0.5, mi_weight=0.5)
    print("挖掘的因子公式:")
    if not rolling_factors:
        print("因子池为空，可能是因子公式无效或数据问题，请检查日志")
        return

    for factor in rolling_factors:
        print(f'{factor}\n')

    # 因子合成
    print("开始因子合成...")
    target = data['return'].shift(-1).values.flatten()
    factors = calculate_all_factors(data, rolling_factors, target_shape=data['return'].shape)
    final_factor, rf_model = synthesize_with_random_forest(factors, target)
    save_results(final_factor, data)

    if rf_model is not None:
        feature_importances = pd.Series(rf_model.feature_importances_, index=[f"Factor_{i+1}" for i in range(len(rolling_factors))])
        print("\n特征重要性（前10个因子）：")
        print(feature_importances.sort_values(ascending=False).head(10))

    # 回测
    print(f"\n开始回测: {backtest_start_date} 至 {backtest_end_date}")
    stock_list = get_hs300_top50(start_date, end_date)
    backtest_synthetic_factor(backtest_start_date, backtest_end_date, stock_list, rolling_factors, rf_model)

if __name__ == "__main__":
    main()