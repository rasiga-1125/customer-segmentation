# suggestion_engine.py
import pandas as pd

def get_marketing_suggestions(profile_df):
    suggestions = []
    for idx, row in profile_df.iterrows():
        if row["Income"] > 0.8 and row["MntWines"] > 500:
            suggestion = "🎁 Promote premium wines and loyalty programs"
        elif row["Income"] < 0 and row.get("Kidhome", 0) > 0.5:
            suggestion = "💸 Offer family-focused deals and discounts"
        elif row.get("NumWebPurchases", 0) > 5:
            suggestion = "💻 Focus on online flash sales"
        else:
            suggestion = "📬 Try personalized email campaigns"
        suggestions.append(suggestion)

    return pd.DataFrame({
        "Cluster": profile_df.index,
        "Suggested Marketing Strategy": suggestions
    })
