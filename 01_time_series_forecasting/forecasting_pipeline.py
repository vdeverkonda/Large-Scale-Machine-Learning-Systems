import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_payment_data(days: int = 730) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    trend = np.linspace(10000, 14000, days)
    weekly = 1500 * np.sin(2 * np.pi * np.arange(days) / 7)
    monthly = 800 * np.sin(2 * np.pi * np.arange(days) / 30)
    noise = np.random.normal(0, 700, days)
    volume = np.maximum(trend + weekly + monthly + noise, 1000)
    avg_ticket = np.random.normal(42, 4, days)
    revenue = volume * avg_ticket * 0.012
    return pd.DataFrame({"date": dates, "volume": volume, "revenue": revenue})


def create_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    for lag in [1, 7, 14, 30]:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    for window in [7, 14, 30]:
        out[f"{target}_rolling_mean_{window}"] = out[target].shift(1).rolling(window).mean()
    return out.dropna()


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def train_forecaster(target="revenue"):
    df = generate_payment_data()
    data = create_features(df, target)
    feature_cols = [c for c in data.columns if c not in ["date", "volume", "revenue"]]
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=RANDOM_SEED)
    model.fit(train[feature_cols], train[target])
    preds = model.predict(test[feature_cols])
    mae, rmse, mape = evaluate(test[target], preds)
    print(f"Target: {target}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    plt.figure(figsize=(11, 5))
    plt.plot(test["date"], test[target], label="Actual")
    plt.plot(test["date"], preds, label="Forecast")
    plt.title(f"Payment {target.title()} Forecast")
    plt.xlabel("Date")
    plt.ylabel(target.title())
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_results.png", dpi=160)
    print("Saved forecast_results.png")


if __name__ == "__main__":
    train_forecaster("revenue")
