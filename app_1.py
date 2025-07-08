import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing import preprocess_data
from clustering import perform_clustering
from suggestion_engine import get_marketing_suggestions

st.set_page_config(page_title="Smart Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation + Marketing Intelligence App")
st.markdown("Upload your **customer data**. We'll clean it, cluster it, and suggest the best marketing plans.")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully ✅")
    df_raw = pd.read_csv(uploaded_file,sep="\t")

    st.subheader("🔍 Step 1: Raw Data Preview")
    st.dataframe(df_raw.head())

    # Preprocessing
    st.subheader("🧹 Step 2: Preprocessing your data")
    df_cleaned, scaled_data = preprocess_data(df_raw)
    st.write("✅ Preprocessing done")

    # Clustering
    st.subheader("🔗 Step 3: Clustering customers")
    k = st.slider("🔢 Select number of clusters", 2, 10, 4)
    df_clustered, cluster_profiles = perform_clustering(df_cleaned, scaled_data, k)
    st.write("✅ Customers clustered into groups")

    # Show Cluster Table
    st.subheader("📊 Cluster Profile Summary")
    st.dataframe(cluster_profiles)

    # Marketing Suggestions
    st.subheader("🎯 Step 4: Marketing Suggestions by Segment")
    marketing_suggestions = get_marketing_suggestions(cluster_profiles)
    st.dataframe(marketing_suggestions)

    # Cluster Visualizations
    st.subheader("📈 Step 5: Visualize Clusters")

    # PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_clustered['PCA1'] = pca_result[:, 0]
    df_clustered['PCA2'] = pca_result[:, 1]
    fig_pca, ax_pca = plt.subplots()
    sns.scatterplot(data=df_clustered, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax_pca)
    ax_pca.set_title("PCA 2D Cluster Visualization")
    st.pyplot(fig_pca)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)
    df_clustered['TSNE1'] = tsne_result[:, 0]
    df_clustered['TSNE2'] = tsne_result[:, 1]
    fig_tsne, ax_tsne = plt.subplots()
    sns.scatterplot(data=df_clustered, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', ax=ax_tsne)
    ax_tsne.set_title("t-SNE Cluster Visualization")
    st.pyplot(fig_tsne)

    # Extra Charts
    st.subheader("📊 Extra Cluster Insights")

    # 1. Bar Chart: Avg TotalSpend by Cluster
    st.markdown("#### 💰 Avg Total Spend by Cluster")
    fig_spend, ax_spend = plt.subplots()
    sns.barplot(data=df_clustered, x="Cluster", y="TotalSpend", estimator='mean', ci=None, ax=ax_spend)
    st.pyplot(fig_spend)

    # 2. Pie Chart: Customer Distribution by Cluster
    st.markdown("#### 👥 Customer Distribution by Cluster")
    cluster_counts = df_clustered['Cluster'].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    # 3. Boxplot: Age Distribution by Cluster
    if 'Age' in df_clustered.columns:
        st.markdown("#### 📦 Age Distribution by Cluster")
        fig_age, ax_age = plt.subplots()
        sns.boxplot(data=df_clustered, x='Cluster', y='Age', ax=ax_age)
        st.pyplot(fig_age)

    # 4. Histogram: Web Visits by Cluster
    if 'NumWebVisitsMonth' in df_clustered.columns:
        st.markdown("#### 🌐 Web Visits Distribution by Cluster")
        fig_web, ax_web = plt.subplots()
        sns.histplot(data=df_clustered, x='NumWebVisitsMonth', hue='Cluster', multiple='stack', bins=8, palette='tab10', ax=ax_web)
        st.pyplot(fig_web)

    # Optional Download
    st.download_button("📥 Download Clustered Data", df_clustered.to_csv(index=False), "clustered_data.csv")

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
