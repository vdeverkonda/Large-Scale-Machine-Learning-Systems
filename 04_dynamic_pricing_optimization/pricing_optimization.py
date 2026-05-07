import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)


def generate_pricing_data(n=500):
    market_index = np.random.normal(0, 1, n)
    price = np.random.uniform(5, 30, n)
    demand = 1400 - 38 * price + 95 * market_index + np.random.normal(0, 80, n)
    demand = np.maximum(demand, 0)
    return pd.DataFrame({"price": price, "market_index": market_index, "demand": demand})


def train_demand_model(df):
    X = df[["price", "market_index"]]
    y = df["demand"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def optimize_price(model, market_index=0.0):
    candidate_prices = np.linspace(5, 30, 200)
    X = pd.DataFrame({"price": candidate_prices, "market_index": market_index})
    predicted_demand = np.maximum(model.predict(X), 0)
    revenue = candidate_prices * predicted_demand
    best_idx = np.argmax(revenue)
    return candidate_prices, predicted_demand, revenue, candidate_prices[best_idx], revenue[best_idx]


def monte_carlo_revenue(model, optimal_price, simulations=5000):
    market = np.random.normal(0, 1, simulations)
    X = pd.DataFrame({"price": np.repeat(optimal_price, simulations), "market_index": market})
    demand = np.maximum(model.predict(X) + np.random.normal(0, 60, simulations), 0)
    revenue = optimal_price * demand
    return revenue


def main():
    df = generate_pricing_data()
    model = train_demand_model(df)
    prices, demand, revenue, best_price, best_revenue = optimize_price(model)
    simulated_revenue = monte_carlo_revenue(model, best_price)
    print(f"Estimated demand equation: demand = {model.intercept_:.2f} + {model.coef_[0]:.2f}*price + {model.coef_[1]:.2f}*market_index")
    print(f"Optimal price: ${best_price:.2f}")
    print(f"Expected revenue at optimal price: ${best_revenue:,.2f}")
    print(f"Monte Carlo mean revenue: ${simulated_revenue.mean():,.2f}")
    print(f"5th percentile revenue: ${np.percentile(simulated_revenue, 5):,.2f}")
    print(f"95th percentile revenue: ${np.percentile(simulated_revenue, 95):,.2f}")
    plt.figure(figsize=(8, 5))
    plt.plot(prices, revenue)
    plt.axvline(best_price, linestyle="--", label=f"Optimal price ${best_price:.2f}")
    plt.title("Revenue Optimization Curve")
    plt.xlabel("Price")
    plt.ylabel("Expected Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pricing_optimization_curve.png", dpi=160)
    print("Saved pricing_optimization_curve.png")


if __name__ == "__main__":
    main()
