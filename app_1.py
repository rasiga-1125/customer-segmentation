import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fpdf import FPDF
import base64

from preprocessing import preprocess_data
from clustering import perform_clustering
from suggestion_engine import get_marketing_suggestions

st.set_page_config(page_title="Smart Customer Segmentation", layout="wide")

# Branding
st.image("https://i.imgur.com/Ob2yZ.png", width=100)
st.markdown("<h2 style='color:#0e76a8;'>AI - Customer Intelligence System</h2>", unsafe_allow_html=True)

st.markdown("Upload your **customer data**. We'll clean it, cluster it, and suggest the best marketing plans.")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully ✅")
    df_raw = pd.read_csv(uploaded_file, sep='\t', engine='python')
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw.apply(pd.to_numeric, errors='ignore')

    st.subheader("🔍 Step 1: Raw Data Preview")
    st.dataframe(df_raw.head())

    st.subheader("🧹 Step 2: Preprocessing your data")
    df_cleaned, scaled_data = preprocess_data(df_raw)
    st.write("✅ Preprocessing done")

    st.subheader("🔧 Step 3: Choose Number of Clusters")
    k = st.slider("How many clusters?", 2, 10, 4)

    st.subheader("🔗 Step 4: Clustering customers")
    df_clustered, cluster_profiles = perform_clustering(df_cleaned, scaled_data, k)
    st.write("✅ Customers clustered into groups")

    st.subheader("📊 Cluster Profile Summary")
    st.dataframe(cluster_profiles)

    st.subheader("🎯 Step 5: Smart Marketing Insights")
    marketing_suggestions = get_marketing_suggestions(cluster_profiles)
    st.dataframe(marketing_suggestions)

    st.subheader("📈 Step 6: Visualize Clusters")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_clustered['PCA1'] = pca_result[:, 0]
    df_clustered['PCA2'] = pca_result[:, 1]
    fig_pca, ax_pca = plt.subplots()
    sns.scatterplot(data=df_clustered, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax_pca)
    st.pyplot(fig_pca)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)
    df_clustered['TSNE1'] = tsne_result[:, 0]
    df_clustered['TSNE2'] = tsne_result[:, 1]
    fig_tsne, ax_tsne = plt.subplots()
    sns.scatterplot(data=df_clustered, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', ax=ax_tsne)
    st.pyplot(fig_tsne)

    st.subheader("📊 Step 7: Extra Cluster Insights")

    # Bar Chart: Avg TotalSpend by Cluster
    if 'TotalSpend' in df_clustered.columns:
        st.markdown("#### 💰 Avg Total Spend by Cluster")
        fig_spend, ax_spend = plt.subplots()
        sns.barplot(data=df_clustered, x="Cluster", y="TotalSpend", estimator='mean', ci=None, ax=ax_spend)
        st.pyplot(fig_spend)

    # Pie Chart: Customer Distribution
    cluster_counts = df_clustered['Cluster'].value_counts()
    st.markdown("#### 👥 Customer Distribution by Cluster")
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)
    fig_pie.savefig("real_cluster_pie.png")


    # PDF Report Generator
    def create_pdf(cluster_profiles):
        from datetime import datetime
        import matplotlib.pyplot as plt
        from fpdf import FPDF
        import os
        import tempfile
        import pandas as pd
    
        pdf = FPDF()
        pdf.add_page()
    
        # Branding Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Customer Segmentation & Marketing Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Prepared by: Rasiga Priya (Pappu)", ln=True, align='C')
        pdf.cell(200, 10, txt="Project Title: Smart Customer Intelligence System", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
    
        # Cluster Profile Details
        if cluster_profiles.empty:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="No cluster data available.", ln=True)
        else:
            for idx, row in cluster_profiles.iterrows():
                pdf.ln(8)
                pdf.set_font("Arial", 'B', size=11)
                pdf.cell(200, 10, txt=f"Cluster {idx}", ln=True)
                pdf.set_font("Arial", size=10)
                for col, val in row.items():
                    val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                    pdf.cell(200, 8, txt=f"{col}: {val_str}", ln=True)
    
        # Insert actual saved pie chart if exists
        if os.path.exists("real_cluster_pie.png"):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Customer Distribution by Cluster", ln=True, align='C')
            pdf.image("real_cluster_pie.png", x=30, w=150)
    
        # Bar chart: Total Spend per cluster
        if 'TotalSpend' in cluster_profiles.columns:
            pdf.add_page()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            cluster_profiles['TotalSpend'] = cluster_profiles['TotalSpend'].astype(float)
            cluster_profiles['TotalSpend'].plot(kind='bar', ax=ax2)
            ax2.set_title("Total Spend by Cluster")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Total Spend (normalized)")
            ax2.set_xticks(range(len(cluster_profiles)))
            ax2.set_xticklabels([f"Cluster {i}" for i in cluster_profiles.index])
            plt.tight_layout()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as barfile:
                fig2.savefig(barfile.name, format="PNG")
                plt.close(fig2)
                pdf.image(barfile.name, x=30, w=150)
    
        # Smart suggestions section
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Smart Marketing Suggestions", ln=True, align='C')
        pdf.set_font("Arial", size=11)
        for idx, row in cluster_profiles.iterrows():
            income = row.get('Income', 0)
            wine = row.get('MntWines', 0)
            web = row.get('NumWebPurchases', 0)
            kids = row.get('Kidhome', 0)
            rec = row.get('Recency', 0)
            if income > 0.8 and wine > 500:
                msg = "Target with premium wine loyalty offers."
            elif income < 0 and kids > 0.5:
                msg = "Promote family-oriented budget bundles."
            elif web > 6:
                msg = "Use aggressive online flash sales."
            elif rec < 10:
                msg = "Push upsell offers to recently active users."
            else:
                msg = "Try personalized email + discount campaign."
            pdf.cell(200, 10, txt=f"Cluster {idx}: {msg}", ln=True)
    
        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, '© 2025 Rasiga Priya | Final Project Submission', 0, 0, 'C')
    
        return pdf.output(dest='S').encode('latin1')


    pdf_bytes = create_pdf(cluster_profiles)
    b64 = base64.b64encode(pdf_bytes).decode()
    st.download_button(
    label="📄 Download Cluster Report PDF",
    data=pdf_bytes,
    file_name="cluster_report.pdf",
    mime="application/pdf"
)

    st.download_button("📥 Download Clustered Data", df_clustered.to_csv(index=False), "clustered_data.csv")

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
