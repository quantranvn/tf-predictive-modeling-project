import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

# Optional statsmodels baselines
HAS_ARIMA = True
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    HAS_ARIMA = False

HAS_SARIMAX = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    HAS_SARIMAX = False

HAS_HW = True
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    HAS_HW = False

# Prophet
HAS_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAS_PROPHET = False

# ML baselines
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

HAS_LGBM = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False

FIG_DIR = Path("figures_test")
OUT_DIR = Path("outputs_test")
FIG_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_and_engineer():
    commits_df = pd.read_csv('preprocessed_tensorflow_commits.csv', parse_dates=['date'])
    issues_df = pd.read_csv('preprocessed_tensorflow_issues.csv', parse_dates=['created_at', 'closed_at'])

    commits_df['month_year'] = commits_df['date'].dt.to_period('M')
    issues_df['month_year'] = issues_df['created_at'].dt.to_period('M')

    contributors_per_month = (
        commits_df.groupby('month_year')['contributor']
        .nunique().reset_index().rename(columns={'contributor': 'contributors_per_month'})
    )

    avg_labels_per_issue = (
        issues_df.groupby('month_year')['labels_count']
        .mean().reset_index().rename(columns={'labels_count': 'avg_labels_count'})
    )

    issues_df['month_year_closed'] = issues_df['closed_at'].dt.to_period('M')
    monthly_closed = (
        issues_df.groupby('month_year_closed')['state']
        .apply(lambda x: (x == 'closed').sum())
        .reset_index().rename(columns={'month_year_closed': 'month_year', 'state': 'monthly_closed_issues'})
    )

    monthly_created = (
        issues_df.groupby('month_year')['id']
        .size().reset_index().rename(columns={'id': 'monthly_created_issues'})
    )

    monthly = pd.merge(monthly_created, monthly_closed, on='month_year', how='outer')
    monthly = pd.merge(monthly, contributors_per_month, on='month_year', how='left')
    monthly = pd.merge(monthly, avg_labels_per_issue, on='month_year', how='left')

    monthly = monthly.sort_values('month_year').reset_index(drop=True)
    monthly[['monthly_created_issues', 'monthly_closed_issues',
             'contributors_per_month', 'avg_labels_count']] = monthly[[
        'monthly_created_issues', 'monthly_closed_issues',
        'contributors_per_month', 'avg_labels_count'
    ]].fillna(0)

    monthly['prev_monthly_closed'] = monthly['monthly_closed_issues'].shift(1)
    monthly['prev_contributors'] = monthly['contributors_per_month'].shift(1)
    monthly['prev_avg_labels'] = monthly['avg_labels_count'].shift(1)

    monthly['date'] = pd.to_datetime(monthly['month_year'].astype(str))
    monthly['quarter'] = monthly['date'].dt.quarter
    monthly['year'] = monthly['date'].dt.year
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['date'].dt.month / 12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['date'].dt.month / 12)

    monthly = monthly.dropna().reset_index(drop=True)

    feature_cols = ['monthly_created_issues', 'prev_monthly_closed', 'prev_contributors',
                    'prev_avg_labels', 'quarter', 'year', 'month_sin', 'month_cos']
    X = monthly[feature_cols].iloc[:-1].reset_index(drop=True)
    y = monthly['monthly_closed_issues'].shift(-1).dropna().reset_index(drop=True)
    timeline = monthly['month_year'].astype(str).iloc[1:].reset_index(drop=True)

    return X, y, timeline, feature_cols


def naive_last_value(y_hist: np.ndarray) -> float:
    return float(y_hist[-1])


def naive_seasonal(y_series: pd.Series, idx: int, timeline: List[str]) -> float:
    target_month = int(pd.to_datetime(timeline[idx]).month)
    past_indices = [i for i in range(idx) if pd.to_datetime(timeline[i]).month == target_month]
    if not past_indices:
        return float(y_series.iloc[idx-1])
    return float(y_series.iloc[past_indices].mean())


def safe_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {'MSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    return {
        'MSE': mean_squared_error(y_true[mask], y_pred[mask]),
        'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
        'R2': r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 1 else np.nan
    }


def _best_sarimax(y_train, seasonal_periods=12):
    if not HAS_SARIMAX:
        return None
    candidates = [
        ((1,1,1), (1,1,1, seasonal_periods)),
        ((0,1,1), (1,1,1, seasonal_periods)),
        ((1,1,0), (1,1,1, seasonal_periods)),
        ((1,0,1), (0,1,1, seasonal_periods)),
    ]
    best_aic = np.inf
    best_model = None
    for order, seasonal_order in candidates:
        try:
            m = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = m.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_model = res
        except Exception:
            continue
    return best_model


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    timeline: pd.Series,
    initial_train: int = 24,
    alpha_from_pkl: float = None,
    use_arima: bool = True,
    use_prophet: bool = True,
    use_sarimax: bool = True,
    use_hw: bool = True,
    use_rf: bool = True,
    use_xgb: bool = True,
    use_lgbm: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

    n = len(y)
    if n <= initial_train + 1:
        raise ValueError(f"Not enough data for walk-forward: n={n}, initial_train={initial_train}")

    rows = []
    y_series = y.copy()

    for t in range(initial_train, n):
        X_train, y_train = X.iloc[:t, :], y.iloc[:t]
        X_test_row = X.iloc[t:t+1, :]
        date_t = timeline.iloc[t]

        scaler_wf = StandardScaler().fit(X_train.values)
        X_train_s = scaler_wf.transform(X_train.values)
        X_test_s = scaler_wf.transform(X_test_row.values)
        if alpha_from_pkl is not None:
            lasso = Lasso(alpha=alpha_from_pkl, max_iter=100000, random_state=42)
        else:
            lasso = Lasso(alpha=0.001, max_iter=100000, random_state=42)
        lasso.fit(X_train_s, y_train.values)
        lasso_pred = float(lasso.predict(X_test_s)[0])

        naive_last_pred = naive_last_value(y_train.values)
        naive_seasonal_pred = naive_seasonal(y_series, t, timeline)

        arima_pred = np.nan
        if HAS_ARIMA and use_arima:
            try:
                model = ARIMA(y_train.values, order=(1,1,1))
                fitted = model.fit(method_kwargs={"warn_convergence": False})
                arima_pred = float(fitted.forecast(steps=1)[0])
            except Exception:
                arima_pred = np.nan

        sarimax_pred = np.nan
        if HAS_SARIMAX and use_sarimax:
            try:
                best_uni = _best_sarimax(y_train.values, seasonal_periods=12)
                if best_uni is not None:
                    sarimax_pred = float(best_uni.forecast(steps=1)[0])
                    sarimax_pred = max(0.0, sarimax_pred)
            except Exception:
                sarimax_pred = np.nan

        hw_pred = np.nan
        if HAS_HW and use_hw:
            try:
                model = ExponentialSmoothing(y_train.values, trend='add', seasonal='add', seasonal_periods=12)
                hw_fit = model.fit(optimized=True)
                hw_pred = float(hw_fit.forecast(1)[0])
                hw_pred = max(0.0, hw_pred)
            except Exception:
                hw_pred = np.nan

        rf_pred = np.nan
        if use_rf:
            try:
                rf = RandomForestRegressor(n_estimators=500, random_state=42)
                rf.fit(X_train.values, y_train.values)
                rf_pred = float(rf.predict(X_test_row.values)[0])
            except Exception:
                rf_pred = np.nan

        xgb_pred = np.nan
        if HAS_XGB and use_xgb:
            try:
                xgb = XGBRegressor(
                    n_estimators=600, learning_rate=0.05, max_depth=5,
                    subsample=0.9, colsample_bytree=0.9, random_state=42
                )
                xgb.fit(X_train.values, y_train.values, verbose=False)
                xgb_pred = float(xgb.predict(X_test_row.values)[0])
            except Exception:
                xgb_pred = np.nan

        lgbm_pred = np.nan
        if HAS_LGBM and use_lgbm:
            try:
                lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42)
                lgbm.fit(X_train.values, y_train.values)
                lgbm_pred = float(lgbm.predict(X_test_row.values)[0])
            except Exception:
                lgbm_pred = np.nan

        prophet_pred = np.nan
        if HAS_PROPHET and use_prophet:
            try:
                df_p = pd.DataFrame({'ds': pd.to_datetime(timeline.iloc[:t]), 'y': y_train.values})
                mprop = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                mprop.fit(df_p)
                future = pd.DataFrame({'ds': [pd.to_datetime(date_t)]})
                fcst = mprop.predict(future)
                prophet_pred = float(fcst['yhat'].iloc[0])
                prophet_pred = max(0.0, prophet_pred)
            except Exception:
                prophet_pred = np.nan

        rows.append({
            'date': date_t,
            'actual': float(y.iloc[t]),
            'lasso_wf': lasso_pred,
            'naive_last': naive_last_pred,
            'naive_seasonal': naive_seasonal_pred,
            'arima': arima_pred,
            'sarimax': sarimax_pred,
            'holt_winters': hw_pred,
            'rf_wf': rf_pred,
            'xgb_wf': xgb_pred,
            'lgbm_wf': lgbm_pred,
            'prophet': prophet_pred
        })

    preds_df = pd.DataFrame(rows)

    metrics = {}
    y_true = preds_df['actual'].values
    for col in ['lasso_wf','naive_last','naive_seasonal','arima','sarimax','holt_winters','rf_wf','xgb_wf','lgbm_wf','prophet']:
        if col in preds_df.columns:
            metrics[col] = safe_metrics(y_true, preds_df[col].values)

    return preds_df, metrics


def evaluate_frozen_model(X, y, timeline, holdout_months=12):
    print(f"== Frozen model evaluation on last {holdout_months} months ==")
    try:
        lasso_cv = joblib.load('predict_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception:
        print("Could not load predict_model.pkl or scaler.pkl. Skipping frozen model evaluation.")
        return None, {}

    n = len(y)
    if n <= holdout_months + 1:
        print(f"Not enough data ({n}) for a {holdout_months}-month hold-out. Skipping.")
        return None, {}

    X_test = X.iloc[-holdout_months:, :]
    y_test = y.iloc[-holdout_months:]
    tl_test = timeline.iloc[-holdout_months:]

    try:
        X_test_s = scaler.transform(X_test.values)
        y_pred = lasso_cv.predict(X_test_s)
        frozen_metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
        }
        frozen_df = pd.DataFrame({
            'date': tl_test.values,
            'actual': y_test.values,
            'frozen_model_pred': y_pred
        })
    except Exception as e:
        print(f"Frozen model evaluation failed: {e}")
        return None, {}

    plt.figure(figsize=(12,6))
    plt.plot(tl_test.values, y_test.values, label='Actual', linewidth=2)
    plt.plot(tl_test.values, y_pred, label='Frozen Lasso (from pkl)', linestyle='--')
    #plt.title(f'Frozen model on held-out last {holdout_months} months')
    plt.xlabel('Month')
    plt.ylabel('Closed Issues (next month)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / f'frozen_model_holdout_{holdout_months}m.svg', format='svg')
    plt.close()

    return frozen_df, frozen_metrics


def plot_predictions(preds_df: pd.DataFrame, title_suffix: str = ""):
    plt.figure(figsize=(16,8))
    plt.plot(preds_df['date'], preds_df['actual'], label='Actual', linewidth=2)
    for col in ['lasso_wf','naive_last','naive_seasonal','arima','sarimax','holt_winters','rf_wf','xgb_wf','lgbm_wf','prophet']:
        if col in preds_df.columns and preds_df[col].notna().any():
            plt.plot(preds_df['date'], preds_df[col], label=col.replace('_',' ').title(), linestyle='--')
    #plt.title(f'Walk-Forward: Actual vs Predictions {title_suffix}'.strip())
    plt.xlabel('Month')
    plt.ylabel('Closed Issues (next month)')
    plt.xticks(rotation=45)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    outfile = FIG_DIR / f'walkforward_actual_vs_predictions{title_suffix.replace(" ", "_")}.svg'
    plt.savefig(outfile, format='svg')
    plt.close()


def plot_metrics_bar(metrics: Dict[str, Dict[str, float]], metric_name: str, title: str, filename: str):
    labels, values = [], []
    for model_name, vals in metrics.items():
        labels.append(model_name)
        values.append(vals.get(metric_name, np.nan))

    plt.figure(figsize=(12,7))
    plt.bar(labels, values)
    #plt.title(title)
    plt.ylabel(metric_name)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(FIG_DIR / filename, format='svg')
    plt.close()


def save_metrics_csv(metrics: Dict[str, Dict[str, float]], filename: str):
    df = pd.DataFrame(metrics).T
    df.to_csv(OUT_DIR / filename, index=True)


def main():
    print("Loading data and engineering features...")
    X, y, timeline, feature_cols = load_and_engineer()
    print(f"Data points for time-aware target: {len(y)} (months)")

    alpha_from_pkl = None
    try:
        lasso_cv = joblib.load('predict_model.pkl')
        if hasattr(lasso_cv, 'alpha_'):
            alpha_from_pkl = float(lasso_cv.alpha_)
            print(f"Loaded alpha from predict_model.pkl: {alpha_from_pkl}")
    except Exception:
        print("Warning: Could not load predict_model.pkl to extract alpha. Using default alpha for walk-forward.")

    frozen_df, frozen_metrics = evaluate_frozen_model(X, y, timeline, holdout_months=12)

    print("Running walk-forward validation (with R^2 plot)...")
    preds_df, metrics = walk_forward_validation(
        X, y, timeline,
        initial_train=24,
        alpha_from_pkl=alpha_from_pkl,
        use_arima=True,
        use_prophet=True,
        use_sarimax=True,
        use_hw=True,
        use_rf=True,
        use_xgb=True,
        use_lgbm=True
    )

    preds_df.to_csv(OUT_DIR / 'timeseries_predictions.csv', index=False)
    if frozen_df is not None:
        frozen_df.to_csv(OUT_DIR / 'frozen_holdout_predictions.csv', index=False)

    if frozen_metrics:
        metrics['frozen_model'] = frozen_metrics

    save_metrics_csv(metrics, 'timeseries_eval_metrics.csv')

    plot_predictions(preds_df, title_suffix="")
    plot_metrics_bar(metrics, 'MSE', 'Walk-Forward Comparison: MSE (lower is better)', 'wf_mse_comparison.svg')
    plot_metrics_bar(metrics, 'MAE', 'Walk-Forward Comparison: MAE (lower is better)', 'wf_mae_comparison.svg')
    plot_metrics_bar(metrics, 'R2', 'Walk-Forward Comparison: R² (higher is better)', 'wf_r2_comparison.svg')

    lasso_resid = preds_df['actual'] - preds_df['lasso_wf']
    arima_resid = preds_df['actual'] - preds_df['arima']
    plt.figure(figsize=(12,6))
    plt.hist(lasso_resid.dropna(), bins=20, alpha=0.6, label='Lasso', color='steelblue', edgecolor='black')
    plt.hist(arima_resid.dropna(), bins=20, alpha=0.6, label='ARIMA', color='orange', edgecolor='black')
    #plt.title('Residual Distributions: Lasso vs ARIMA (Walk-Forward)')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(FIG_DIR / 'residuals_lasso_vs_arima.svg', format='svg')
    plt.close()

    print("\n=== Summary ===")
    print("Metrics written to outputs/timeseries_eval_metrics.csv")
    print("Predictions written to outputs/timeseries_predictions.csv")
    print(f"Plots saved under {FIG_DIR}/")

if __name__ == "__main__":
    main()
