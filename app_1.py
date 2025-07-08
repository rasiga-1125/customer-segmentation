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
st.markdown("<h2 style='color:#0e76a8;'>Pappu AI - Customer Intelligence System</h2>", unsafe_allow_html=True)

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

    # PDF Report Generator
    def create_pdf(cluster_profiles):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Cluster Profile Report", ln=True, align='C')
        for idx, row in cluster_profiles.iterrows():
            pdf.ln(10)
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(200, 10, txt=f"Cluster {idx}", ln=True)
            pdf.set_font("Arial", size=10)
            for col, val in row.items():
                pdf.cell(200, 8, txt=f"{col}: {round(val,2)}", ln=True)
        return pdf.output(dest='S').encode('latin1')

    pdf_bytes = create_pdf(cluster_profiles)
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(f"📄 [Download Cluster Report PDF](data:application/pdf;base64,{b64})", unsafe_allow_html=True)

    st.download_button("📥 Download Clustered Data", df_clustered.to_csv(index=False), "clustered_data.csv")

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
