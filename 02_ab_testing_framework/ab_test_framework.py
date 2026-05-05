import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


def simulate_experiment(n_control=12000, n_treatment=12000):
    control_conversion_rate = 0.082
    treatment_conversion_rate = 0.091
    control = np.random.binomial(1, control_conversion_rate, n_control)
    treatment = np.random.binomial(1, treatment_conversion_rate, n_treatment)
    control_revenue = control * np.random.gamma(shape=4, scale=18, size=n_control)
    treatment_revenue = treatment * np.random.gamma(shape=4.2, scale=18.5, size=n_treatment)
    return pd.DataFrame({
        "group": ["control"] * n_control + ["treatment"] * n_treatment,
        "converted": np.concatenate([control, treatment]),
        "revenue": np.concatenate([control_revenue, treatment_revenue]),
    })


def two_proportion_z_test(control_success, treatment_success, control_n, treatment_n):
    p_pool = (control_success + treatment_success) / (control_n + treatment_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (treatment_success / treatment_n - control_success / control_n) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def bootstrap_ci(control_values, treatment_values, n_bootstrap=3000):
    diffs = []
    for _ in range(n_bootstrap):
        c = np.random.choice(control_values, size=len(control_values), replace=True)
        t = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
        diffs.append(t.mean() - c.mean())
    return np.percentile(diffs, [2.5, 97.5])


def main():
    df = simulate_experiment()
    summary = df.groupby("group").agg(
        users=("converted", "count"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"),
        revenue_per_user=("revenue", "mean"),
        total_revenue=("revenue", "sum"),
    )
    print(summary)
    c = df[df.group == "control"]
    t = df[df.group == "treatment"]
    z, p = two_proportion_z_test(c.converted.sum(), t.converted.sum(), len(c), len(t))
    revenue_ci = bootstrap_ci(c.revenue.values, t.revenue.values)
    conversion_lift = (t.converted.mean() / c.converted.mean() - 1) * 100
    revenue_lift = (t.revenue.mean() / c.revenue.mean() - 1) * 100
    projected_users = 1_000_000
    projected_incremental_revenue = (t.revenue.mean() - c.revenue.mean()) * projected_users
    print(f"\nConversion lift: {conversion_lift:.2f}%")
    print(f"Revenue per user lift: {revenue_lift:.2f}%")
    print(f"Two-proportion z-test: z={z:.3f}, p={p:.5f}")
    print(f"95% CI for revenue/user lift: [{revenue_ci[0]:.4f}, {revenue_ci[1]:.4f}]")
    print(f"Projected incremental revenue per 1M users: ${projected_incremental_revenue:,.2f}")
    plt.figure(figsize=(7, 5))
    plt.bar(summary.index, summary["revenue_per_user"])
    plt.title("Revenue Per User by Experiment Group")
    plt.ylabel("Revenue Per User")
    plt.tight_layout()
    plt.savefig("ab_test_revenue.png", dpi=160)
    print("Saved ab_test_revenue.png")


if __name__ == "__main__":
    main()
