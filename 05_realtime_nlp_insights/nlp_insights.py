import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

MESSAGES = [
    "My international wire transfer is delayed and the beneficiary bank has not received funds",
    "I need help changing transaction limits for corporate payments",
    "The FX conversion rate was different from what I expected",
    "Payment was rejected because beneficiary details were incorrect",
    "Can you explain the product fee and interest rate on this account",
    "I cannot find the payment status for a supplier transfer",
    "The foreign currency conversion happened at the receiving bank and caused extra fees",
    "Our treasury team needs higher daily payment limits",
    "The payment failed after account verification",
    "I need documentation about pricing and product servicing",
    "Wire payment status is unclear and customer needs urgent confirmation",
    "We want better FX pricing before payment execution",
    "The beneficiary bank rejected the transfer due to missing account information",
    "Need support for onboarding payment products and account setup",
    "Client is asking about interest rate changes and balance behavior",
]

RISK_TERMS = {"delayed", "rejected", "failed", "urgent", "missing", "unclear", "extra fees"}


def risk_score(message: str) -> int:
    lower = message.lower()
    return sum(term in lower for term in RISK_TERMS)


def main():
    df = pd.DataFrame({"message": MESSAGES})
    df["risk_score"] = df["message"].apply(risk_score)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["message"])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["topic_cluster"] = kmeans.fit_predict(X)
    terms = vectorizer.get_feature_names_out()
    print("Cluster Themes:")
    for cluster_id in sorted(df["topic_cluster"].unique()):
        center = kmeans.cluster_centers_[cluster_id]
        top_terms = [terms[i] for i in center.argsort()[-6:][::-1]]
        print(f"Cluster {cluster_id}: {', '.join(top_terms)}")
    print("\nHigh Priority Messages:")
    print(df[df["risk_score"] >= 1].sort_values("risk_score", ascending=False)[["message", "topic_cluster", "risk_score"]])
    df.to_csv("nlp_insights_output.csv", index=False)
    print("\nSaved nlp_insights_output.csv")


if __name__ == "__main__":
    main()
