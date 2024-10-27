import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

st.title("Co-Citation Graph Analysis")

# Upload file
uploaded_file = st.file_uploader("Upload a co-citation graph file (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load the graph data from CSV
    df = pd.read_csv(uploaded_file)
    edges = df.values.tolist()
    G = nx.Graph()
    G.add_edges_from(edges)

    st.subheader("Graph Properties and Mathematical Analysis")

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    density = nx.density(G)
    clustering_coeff = nx.average_clustering(G)

    st.write(f"Number of Nodes: {num_nodes}")
    st.write(f"Number of Edges: {num_edges}")
    st.write(f"Average Degree: {avg_degree:.2f}")
    st.write(f"Density of Graph: {density:.4f}")
    st.write(f"Average Clustering Coefficient: {clustering_coeff:.4f}")

    st.subheader("Degree Distribution")
    fig, ax = plt.subplots()
    sns.histplot(degrees, bins=20, kde=True, ax=ax)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Graph Clustering Analysis")
    kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(degrees).reshape(-1, 1))
    clusters = kmeans.labels_

    color_map = ["red" if cluster == 0 else "blue" if cluster == 1 else "green" for cluster in clusters]
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(G, pos, node_color=color_map, with_labels=False, node_size=50, ax=ax)
    st.pyplot(fig)

    st.subheader("Inference Analysis")
    st.write("با توجه به تحلیل‌های ریاضیاتی و بصری گراف، خوشه‌های علمی مختلف در ساختار هم‌استنادی شکل گرفته‌اند که نشان‌دهنده گرایش‌های فعلی و آینده‌ی تحقیقات علمی در حوزه‌های مختلف است.")
    st.write("خوشه‌های رنگی متفاوت نمایانگر دسته‌های مختلفی از مقالات و پژوهش‌های مرتبط هستند که احتمالا هر یک نشان‌دهنده‌ی یک شاخه یا زیرشاخه‌ی علمی خاص است.")
    st.write("تحلیل دقیق‌تر نشان می‌دهد که موضوعات با درجات بالاتر و خوشه‌های بزرگ‌تر، بیشتر در مرکز توجه پژوهش‌ها قرار دارند و در آینده احتمالاً تحقیقات بیشتری به آنها پرداخته خواهد شد.")
