# suggestion_engine.py
import pandas as pd

def get_marketing_suggestions(profile_df):
    suggestions = []
    for idx, row in profile_df.iterrows():
        income = row.get('Income', 0)
        wines = row.get('MntWines', 0)
        web = row.get('NumWebPurchases', 0)
        kids = row.get('Kidhome', 0)
        recency = row.get('Recency', 0)

        if income > 0.8 and wines > 500:
            insight = f"Cluster {idx} represents high-income wine lovers. Use premium loyalty offers."
        elif income < 0 and kids > 0.5:
            insight = f"Cluster {idx} includes family-focused budget shoppers. Promote bundle deals."
        elif web > 6:
            insight = f"Cluster {idx} is digitally active. Focus on personalized emails and online flash sales."
        elif recency < 10:
            insight = f"Cluster {idx} has recently engaged customers. Great for upselling."
        else:
            insight = f"Cluster {idx} is moderately engaged. Try reactivation campaigns."

        suggestions.append(insight)
    return pd.DataFrame({'Cluster': profile_df.index, 'Smart Insight': suggestions})
