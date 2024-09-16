import streamlit as st 

import pandas as pd 
import joblib 
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder




st.title("International football results from 1872 to 2024")

# pip install -r C:\Users\Admin\OneDrive\Desktop\ML\ml_Football\requirements.txt
# python -m streamlit run C:\Users\Admin\OneDrive\Desktop\ML\ml_Football\deployment.py

combined_df_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/combined_df.csv"
# Read the CSV file into a DataFrame
combined_df = pd.read_csv(combined_df_path)

st.write("Dataset records:")
st.write(combined_df)

# Load the LabelEncoder from the file
loaded_label_encoder_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/label_encoder.pkl"
loaded_label_encoder = joblib.load(loaded_label_encoder_path)

# Load the StandardScaler from the file
loaded_scaler_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/standard_scaler.pkl"
loaded_scaler = joblib.load(loaded_scaler_path)


# Fit and transform the specified column
combined_df['Team'] = loaded_label_encoder.fit_transform(combined_df['Team'])

# Select numerical features for scaling
numerical_features = combined_df.select_dtypes(include=np.number).columns

# Fit the scaler on the numerical features and transform the data
combined_df[numerical_features] = loaded_scaler.fit_transform(combined_df[numerical_features])

# st.write(combined_df)

def perform_pca_and_plot(df):
    pca_features = ['Home_Total_Score', 'Home_Count', 'Home_Average_Score',
                    'Away_Total_Score', 'Away_Count', 'Away_Average_Score',
                    'Number of Goals', 'Sum_First_Goal_Time', 'First_Goal_Count',
                    'Average_First_Goal_Time', 'True_Neutral_Venue', 'False_Neutral_Venue']

    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[pca_features])

    # Create a new DataFrame with the principal components
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Team_encoded'] = df['Team'].values  # Add the label-encoded 'Team' column

    # Visualize the PCA result
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue='Team_encoded', palette='viridis')
    plt.title('PCA of Football Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Use Streamlit to display the plot
    st.pyplot(plt)

    # Optional: Add text or other Streamlit elements
    st.write("This is a PCA scatter plot")

    # Return the DataFrame containing t-SNE results
    return df_pca

# Perform t-SNE on the numeric features
def perform_tsne_and_plot(df):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_features = ['Home_Total_Score', 'Home_Count', 'Home_Average_Score',
                    'Away_Total_Score', 'Away_Count', 'Away_Average_Score',
                    'Number of Goals', 'Sum_First_Goal_Time', 'First_Goal_Count',
                    'Average_First_Goal_Time', 'True_Neutral_Venue', 'False_Neutral_Venue']
    tsne_result = tsne.fit_transform(df[tsne_features])  # Apply t-SNE only on numeric features

    # Create a DataFrame for the t-SNE result
    df_tsne = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
    df_tsne['Team_encoded'] = df['Team'].values  # Add the label-encoded 'Team' column

    # Visualize the t-SNE result
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df_tsne, hue='Team_encoded', palette='coolwarm')
    plt.title('t-SNE of Football Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Use Streamlit to display the plot
    st.pyplot(plt)

    # Optional: Add text or other Streamlit elements
    st.write("This is a t-SNE scatter plot")

    # Return the DataFrame containing t-SNE results
    return df_tsne

# Define options for the radio button
dim_reduction_opt = ['PCA', 't-SNE']
# Create a radio button widget
dim_selected_option = st.radio('Choose an option:', dim_reduction_opt)
# Display the selected option
st.write(f'Dispalying: {dim_selected_option} scatter plot')

# Based on the selected option, generate the corresponding plot
if dim_selected_option == 'PCA':
    final_dim_val = perform_pca_and_plot(combined_df)
    # st.write(final_dim_val)
    

elif dim_selected_option == 't-SNE':
    final_dim_val =  perform_tsne_and_plot(combined_df)
    # st.write(final_dim_val)


# Define options for the dropdown menu
clusteing_options = ['K-Means', 'MeanShift', 'Hierarchical', 
                    'DBSCAN', 'Gaussian Mixture Models', 
                    'Spectral Clustering', 'Birch']

# Create the dropdown menu using st.selectbox
clustering_selected_option = st.selectbox('Choose an Clustering Option:', clusteing_options)

# Display the selected option
st.write(f'You selected: {clustering_selected_option}')

# K-Means PCA
def perform_kmeans_PCA(final_dim_val):
    kmeans_pca_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    
    kmeans_pca = KMeans(n_clusters=kmeans_pca_slider, random_state=42, n_init=10)

    # Fits the K-Means model on the dataset with the first two principal components (PC1 and PC2)
    final_dim_val['KMeans_Cluster_PCA'] = kmeans_pca.fit_predict(final_dim_val[['PC1', 'PC2']])
    # Visualize the K-Means clustering result with PCA

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='KMeans_Cluster_PCA', palette='viridis')

    # Plot the centroids
    centroids = kmeans_pca.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')

    # Title and legend
    plt.title(f'K-Means Clustering with PCA (Cluster number = {kmeans_pca_slider})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters")  # Use ncol for multiple columns

    st.pyplot(plt)
    


# K-Menas t-SNE
def perform_kmeans_TSNE(final_dim_val): 
    kmeans_tsne_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    
    kmeans_tsne = KMeans(n_clusters=kmeans_tsne_slider, random_state=42, n_init=10)
    kmeans_tsne.fit(final_dim_val[['t-SNE1', 't-SNE2']])

    # Fits the K-Means model on the dataset with the first two principal components (t-SNE1 and t-SNE2)
    final_dim_val['KMeans_Cluster_TSNE'] = kmeans_tsne.labels_

    # Visualize the K-Means clustering result with TSNE
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='KMeans_Cluster_TSNE', palette='viridis')

    # Plot the centroids
    centroids = kmeans_tsne.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')

    # Title and legend
    plt.title(f'K-Means Clustering with t-SNE (Cluster number = {kmeans_tsne_slider})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)

# MeanShift PCA
def perform_meanshift_PCA(final_dim_val):
    ms_pca_bandwidth_val = st.slider('Select a bandwidth value (2.0 is recommended):', min_value=0.1, max_value=5.0, step=0.01)
    meanshift_pca = MeanShift(bandwidth=ms_pca_bandwidth_val)
    meanshift_pca.fit(final_dim_val[['PC1', 'PC2']])
    final_dim_val['MeanShift_Cluster_PCA'] = meanshift_pca.labels_

    # Visualize the MeanShift clustering result with PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='MeanShift_Cluster_PCA', palette='viridis')

    # Plot the centroids
    cluster_centers = meanshift_pca.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', label='Centroids', marker='X')

    # Title and legend
    plt.title(f'MeanShift Clustering with PCA (Bandwidth = {ms_pca_bandwidth_val})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)



# MeanShift t-SNE
def perform_meanshift_TSNE(final_dim_val):
    ms_tsne_bandwidth_val = st.slider('Select a bandwidth value (2.5 is recommended):', min_value=0.1, max_value=5.0, step=0.01)
    meanshift_tsne = MeanShift(bandwidth=ms_tsne_bandwidth_val)
    meanshift_tsne.fit(final_dim_val[['t-SNE1', 't-SNE2']])
    final_dim_val['MeanShift_Cluster_TSNE'] = meanshift_tsne.labels_

    # Visualize the MeanShift clustering result with PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='MeanShift_Cluster_TSNE', palette='viridis')

    # Plot the centroids
    cluster_centers = meanshift_tsne.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', label='Centroids', marker='X')

    # Title and legend
    plt.title(f'MeanShift Clustering with TSNE (Bandwidth = {ms_tsne_bandwidth_val})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)


# # Hierarchical PCA
def perform_hierarchical_PCA(final_dim_val): 
    hierarchical_pca_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=6)
    hierarchical_pca = AgglomerativeClustering(n_clusters=hierarchical_pca_slider, affinity='euclidean', linkage='ward')
    final_dim_val['Hierarchical_Cluster_Complete_PCA'] = hierarchical_pca.fit_predict(final_dim_val[['PC1', 'PC2']])

    # Calculate centroids for each cluster
    centroids = final_dim_val.groupby('Hierarchical_Cluster_Complete_PCA')[['PC1', 'PC2']].mean()

    # Plot the hierarchical clustering result with centroids
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='Hierarchical_Cluster_Complete_PCA', palette='viridis', legend='full')

    # Plot centroids
    plt.scatter(centroids['PC1'], centroids['PC2'], s=200, c='red', label='Centroids', marker='X')

    # Plot title and display
    plt.title(f'Hierarchical Clustering with PCA (Optimal Clusters = {hierarchical_pca_slider})')
    plt.legend()
    st.pyplot(plt)




# # Hierarchical t-SNE
def perform_hierarchical_TSNE(final_dim_val):
    hierarchical_tsne_slider = st.slider('Select a value for number of cluster (9 is recommended):', min_value=2, max_value=6) 
    # Apply AgglomerativeClustering with the optimal number of clusters
    hierarchical_tsne = AgglomerativeClustering(n_clusters=hierarchical_tsne_slider, affinity='euclidean', linkage='ward')
    final_dim_val['Hierarchical_Cluster_Complete_TSNE'] = hierarchical_tsne.fit_predict(final_dim_val[['t-SNE1', 't-SNE2']])

    # Calculate centroids for each cluster
    centroids_tsne = final_dim_val.groupby('Hierarchical_Cluster_Complete_TSNE')[['t-SNE1', 't-SNE2']].mean()

    # Visualize the hierarchical clustering result with centroids
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='Hierarchical_Cluster_Complete_TSNE', palette='viridis', legend='full')

    # Plot centroids (red X markers)
    plt.scatter(centroids_tsne['t-SNE1'], centroids_tsne['t-SNE2'], s=200, c='red', label='Centroids', marker='X')

    # Add title and show the plot
    plt.title(f'Hierarchical Clustering with t-SNE (Optimal Clusters = {hierarchical_tsne_slider})')
    plt.legend()
    st.pyplot(plt)



# # DBSCAN PCA
def perform_dbscan_PCA(final_dim_val):
    dbscan_pca_val = st.slider('Select a eps value (0.77 is recommended):', min_value=0.1, max_value=5.0, step=0.01)
    

    # Step 5: Determine the best min_samples by evaluating silhouette score
    min_samples_range = range(2, 10)
    silhouette_scores = []

    for min_samples in min_samples_range:
        dbscan_pca = DBSCAN(eps=dbscan_pca_val, min_samples=min_samples)
        labels = dbscan_pca.fit_predict(final_dim_val[['PC1', 'PC2']])

        # Calculate silhouette score if more than one cluster is found and not all are noise
        if len(np.unique(labels)) > 1 and np.any(labels != -1):
            silhouette_avg = silhouette_score(final_dim_val[['PC1', 'PC2']], labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)  # Invalid silhouette score if only one cluster/noise
    
    # Step 6: Find the best min_samples based on silhouette score
    best_min_samples = min_samples_range[np.argmax(silhouette_scores)]

    # Step 7: Apply DBSCAN with the optimal min_samples and eps
    dbscan_pca = DBSCAN(eps=dbscan_pca_val, min_samples=best_min_samples)
    final_dim_val['DBSCAN_Cluster_PCA'] = dbscan_pca.fit_predict(final_dim_val[['PC1', 'PC2']])

    # Step 7.5: Calculate the centroids of the clusters (excluding noise points, labeled as -1)
    centroids = final_dim_val[final_dim_val['DBSCAN_Cluster_PCA'] != -1].groupby('DBSCAN_Cluster_PCA')[['PC1', 'PC2']].mean()

    # Step 8: Visualize the DBSCAN clustering result with PCA
    plt.figure(figsize=(10, 7))

    # Scatter plot of DBSCAN clusters
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='DBSCAN_Cluster_PCA', palette='viridis', legend='full')

    # Plot the centroids as red "X" markers
    plt.scatter(centroids['PC1'], centroids['PC2'], s=300, c='red', marker='X', label='Centroids')

    # Title and automatic legend handling by seaborn
    plt.title(f'DBSCAN Clustering with PCA (eps = {dbscan_pca_val}, min_samples = {best_min_samples})')

    # Display legend with cluster labels and centroids
    plt.legend()
    st.pyplot(plt)


# # DBSCAN t-SNE
def perform_dbscan_TSNE(final_dim_val):
    dbscan_tsne_val = st.slider('Select a eps value (2.17 is recommended):', min_value=0.1, max_value=5.0, step=0.01)

    # Step 5: Determine the best min_samples by evaluating silhouette score
    min_samples_range = range(2, 10)
    silhouette_scores = []

    for min_samples in min_samples_range:
        dbscan_tsne = DBSCAN(eps=dbscan_tsne_val, min_samples=min_samples)
        labels = dbscan_tsne.fit_predict(final_dim_val[['t-SNE1', 't-SNE2']])

        # Calculate silhouette score if more than one cluster is found and not all are noise
        if len(np.unique(labels)) > 1 and np.any(labels != -1):
            silhouette_avg = silhouette_score(final_dim_val[['t-SNE1', 't-SNE2']], labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)  # Invalid silhouette score if only one cluster/noise

    # Step 6: Find the best min_samples based on silhouette score
    best_min_samples = min_samples_range[np.argmax(silhouette_scores)]
    best_silhouette_score = max(silhouette_scores)

    print(f"Best min_samples: {best_min_samples}, Best silhouette score: {best_silhouette_score}")

    # Step 7: Apply DBSCAN with the optimal min_samples and eps
    dbscan_tsne = DBSCAN(eps=dbscan_tsne_val, min_samples=best_min_samples)
    final_dim_val['DBSCAN_Cluster_TSNE'] = dbscan_tsne.fit_predict(final_dim_val[['t-SNE1', 't-SNE2']])

    # Step 7.5: Calculate the centroids of the clusters (excluding noise points, labeled as -1)
    centroids = final_dim_val[final_dim_val['DBSCAN_Cluster_TSNE'] != -1].groupby('DBSCAN_Cluster_TSNE')[['t-SNE1', 't-SNE2']].mean()

    # Step 8: Visualize the DBSCAN clustering result with t-SNE
    plt.figure(figsize=(10, 7))

    # Scatter plot of DBSCAN clusters
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='DBSCAN_Cluster_TSNE', palette='viridis', legend='full')

    # Plot the centroids as red "X" markers
    plt.scatter(centroids['t-SNE1'], centroids['t-SNE2'], s=300, c='red', marker='X', label='Centroids')

    # Title and automatic legend handling by seaborn
    plt.title(f'DBSCAN Clustering with t-SNE (eps = {dbscan_tsne_val}, min_samples = {best_min_samples})')

    # Display legend with cluster labels and centroids
    plt.legend()
    st.pyplot(plt)


# # Gaussian Mixture Models PCA
def perform_gmm_PCA(final_dim_val): 
    gmm_pca_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=8)
    # Apply GMM clustering with the optimal number of clusters found
    gmm_pca = GaussianMixture(n_components=gmm_pca_slider, random_state=42)
    final_dim_val['GMM_Cluster_BIC_PCA'] = gmm_pca.fit_predict(final_dim_val[['PC1', 'PC2']])

    # Visualize the GMM clustering result with PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='GMM_Cluster_BIC_PCA', palette='viridis')

    # Plot the cluster means (equivalent to centroids in K-Means)
    means = gmm_pca.means_
    plt.scatter(means[:, 0], means[:, 1], s=300, c='red', label='Cluster Means', marker='X')

    # Title and legend
    plt.title(f'GMM Clustering with PCA (Optimal k = {gmm_pca_slider})')
    plt.legend()
    st.pyplot(plt)


# # Gaussian Mixture Models t-SNE
def perform_gmm_TSNE(final_dim_val): 
    gmm_pca_slider = st.slider('Select a value for number of cluster (5 is recommended):', min_value=2, max_value=8)
    # Apply GMM with the optimal number of components
    gmm_tsne = GaussianMixture(n_components=gmm_pca_slider, random_state=42) # Use optimal_n_components instead of optimal_n_components_tsne
    gmm_tsne.fit(final_dim_val[['t-SNE1', 't-SNE2']])

    # Add the cluster labels to the DataFrame
    final_dim_val['GMM_Cluster_BIC_TSNE'] = gmm_tsne.predict(final_dim_val[['t-SNE1', 't-SNE2']])

    # Visualize the GMM clustering result
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='GMM_Cluster_BIC_TSNE', palette='viridis')

    # Plot the means of the Gaussian components
    means = gmm_tsne.means_
    plt.scatter(means[:, 0], means[:, 1], s=300, c='red', label='Means', marker='X')

    # Title and legend
    plt.title(f'GMM Clustering with t-SNE (Optimal n_components = {gmm_pca_slider})') # Use optimal_n_components in the title as well
    plt.legend()
    st.pyplot(plt)


# Spectral Clustering PCA
def perform_spectral_PCA(final_dim_val):
    spectral_pca_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    spectral_optimal = SpectralClustering(n_clusters=spectral_pca_slider, random_state=42)
    final_dim_val['Spectral_Cluster'] = spectral_optimal.fit_predict(final_dim_val[['PC1', 'PC2']])

    # Calculate centroids as the mean of the points in each cluster
    centroids = final_dim_val.groupby('Spectral_Cluster')[['PC1', 'PC2']].mean().reset_index()
    centroids.rename(columns={'PC1': 'Centroid_PC1', 'PC2': 'Centroid_PC2'}, inplace=True)

    # Merge centroids with the original DataFrame for easy plotting
    final_results_pca_with_centroids = final_dim_val.merge(centroids, left_on='Spectral_Cluster', right_on='Spectral_Cluster', how='left')

    # Visualize the Spectral Clustering result with PCA and centroids
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='Spectral_Cluster', palette='viridis', marker='o', s=50, alpha=0.7)
    sns.scatterplot(x='Centroid_PC1', y='Centroid_PC2', data=centroids, color='red', s=200, marker='X', label='Centroids')
    plt.title(f'Spectral Clustering with PCA (Cluster Number = {spectral_pca_slider})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)



# Spectral Clustering t-SNE
def perform_spectral_TSNE(final_dim_val): 
    spectral_tsne_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    spectral_optimal = SpectralClustering(n_clusters=spectral_tsne_slider, random_state=42)
    final_dim_val['Spectral_Cluster_TSNE'] = spectral_optimal.fit_predict(final_dim_val[['t-SNE1', 't-SNE2']])

    # Calculate centroids as the mean of the points in each cluster
    centroids = final_dim_val.groupby('Spectral_Cluster_TSNE')[['t-SNE1', 't-SNE2']].mean().reset_index()
    centroids.rename(columns={'t-SNE1': 'Centroid_t-SNE1', 't-SNE2': 'Centroid_t-SNE2'}, inplace=True)

    # Visualize the Spectral Clustering result with t-SNE and centroids
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='Spectral_Cluster_TSNE', palette='viridis', marker='o', s=50, alpha=0.7)
    sns.scatterplot(x='Centroid_t-SNE1', y='Centroid_t-SNE2', data=centroids, color='red', s=200, marker='X', label='Centroids')
    plt.title(f'Spectral Clustering with t-SNE (Cluster Number = {spectral_tsne_slider})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)

# Birch PCA
def perform_birch_PCA(final_dim_val): 
    birch_pca_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    birch_pca = Birch(n_clusters=birch_pca_slider)  # Use the optimal number of clusters from previous code
    final_dim_val['Birch_Cluster_PCA'] = birch_pca.fit_predict(final_dim_val[['PC1', 'PC2']])

    # Calculate centroids as the mean of the points in each cluster
    centroids = final_dim_val.groupby('Birch_Cluster_PCA')[['PC1', 'PC2']].mean().reset_index()
    centroids.rename(columns={'PC1': 'Centroid_PC1', 'PC2': 'Centroid_PC2'}, inplace=True)

    # Merge centroids with the original DataFrame for easy plotting
    final_results_pca_with_centroids = final_dim_val.merge(centroids, left_on='Birch_Cluster_PCA', right_on='Birch_Cluster_PCA', how='left')

    # Visualize the Birch clustering result with PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=final_dim_val, hue='Birch_Cluster_PCA', palette='viridis', marker='o')
    sns.scatterplot(x='Centroid_PC1', y='Centroid_PC2', data=centroids, color='red', s=100, marker='X', label='Centroids')
    plt.title(f'Birch Clustering with PCA (Cluster Number = {birch_pca_slider})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)

# Birch t-SNE
def perform_birch_TSNE(final_dim_val):
    birch_tsne_slider = st.slider('Select a value for number of cluster (2 is recommended):', min_value=2, max_value=10)
    birch_tsne = Birch(n_clusters=birch_tsne_slider)  # Adjust n_clusters as needed
    final_dim_val['Birch_Cluster_TSNE'] = birch_tsne.fit_predict(final_dim_val[['t-SNE1', 't-SNE2']])

    # Calculate centroids as the mean of the points in each cluster
    centroids = final_dim_val.groupby('Birch_Cluster_TSNE')[['t-SNE1', 't-SNE2']].mean().reset_index()
    centroids.rename(columns={'t-SNE1': 'Centroid_t-SNE1', 't-SNE2': 'Centroid_t-SNE2'}, inplace=True)

    # Visualize the Birch clustering result with t-SNE
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=final_dim_val, hue='Birch_Cluster_TSNE', palette='viridis', marker='o')

    # Plot the centroids
    plt.scatter(x=centroids['Centroid_t-SNE1'], y=centroids['Centroid_t-SNE2'], color='red', s=100, marker='X', label='Centroids')

    plt.title('Birch Clustering with t-SNE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, title="Clusters") 

    st.pyplot(plt)

# Use if-elif-else to call the appropriate function based on selected option
if clustering_selected_option == 'K-Means' and dim_selected_option == 'PCA':
    perform_kmeans_PCA(final_dim_val)

elif clustering_selected_option == 'K-Means' and dim_selected_option == 't-SNE':
    perform_kmeans_TSNE(final_dim_val)

elif clustering_selected_option == 'MeanShift' and dim_selected_option == 'PCA':
    perform_meanshift_PCA(final_dim_val)

elif clustering_selected_option == 'MeanShift' and dim_selected_option == 't-SNE':
    perform_meanshift_TSNE(final_dim_val)

elif clustering_selected_option == 'Hierarchical' and dim_selected_option == 'PCA':
    perform_hierarchical_PCA(final_dim_val)

elif clustering_selected_option == 'Hierarchical' and dim_selected_option == 't-SNE':
    perform_hierarchical_TSNE(final_dim_val)

elif clustering_selected_option == 'DBSCAN' and dim_selected_option == 'PCA':
    perform_dbscan_PCA(final_dim_val)

elif clustering_selected_option == 'DBSCAN' and dim_selected_option == 't-SNE':
    perform_dbscan_TSNE(final_dim_val)

elif clustering_selected_option == 'Gaussian Mixture Models' and dim_selected_option == 'PCA':
    perform_gmm_PCA(final_dim_val)

elif clustering_selected_option == 'Gaussian Mixture Models' and dim_selected_option == 't-SNE':
    perform_gmm_TSNE(final_dim_val)

elif clustering_selected_option == 'Spectral Clustering' and dim_selected_option == 'PCA':
    perform_spectral_PCA(final_dim_val)

elif clustering_selected_option == 'Spectral Clustering' and dim_selected_option == 't-SNE':
    perform_spectral_TSNE(final_dim_val)

elif clustering_selected_option == 'Birch' and dim_selected_option == 'PCA':
    perform_birch_PCA(final_dim_val)

elif clustering_selected_option == 'Birch' and dim_selected_option == 't-SNE':
    perform_birch_TSNE(final_dim_val)

else:
    st.write("Please select a clustering option.")


# Display a thicker red horizontal line using HTML and CSS
st.markdown("""
    <hr style="border: 1px solid pink;">
""", unsafe_allow_html=True)


# Define a function to handle form input and calculations
st.title("Clustering Deployment")
def get_user_input():
    with st.form(key='football_data_form'):
        # Collect user input from form
        team_name = st.text_input("Enter the team name:", value="LegendTeam")
        home_total_score = st.number_input('Enter Total Home Score', min_value=0, step=1, value=573)
        home_count = st.number_input('Enter Number Of Home Count', min_value=0, step=1, value=315)
        away_total_score = st.number_input('Enter Total Away Score', min_value=0, step=1, value=377)
        away_count = st.number_input('Enter Number Of Away Count', min_value=0, step=1, value=208)
        number_of_goals = st.number_input('Enter Total Number of Goals', min_value=0, step=1, value=624)
        sum_first_goal_time = st.number_input('Enter the Sum of First Goal Time', format="%.2f", value=11383.00)
        first_goal_count = st.number_input('Enter the Number Of First Goal Count', min_value=0, step=1, value=313)
        true_neutral_venue = st.number_input('Enter the Number Of Neutral Venue Matches', min_value=1, step=1, value=183)
        false_neutral_venue = st.number_input('Enter the Number Of NOT Neutral Matches', min_value=0, step=1, value=172)

        submit_button = st.form_submit_button(label='Submit')

        # Inject custom CSS to style the submit button
        st.markdown("""
            <style>
            .stButton>button {
                background-color: transparent; /* No color */
                color: #4CAF50; /* Green font */
                border: 2px solid #4CAF50; /* Green border */
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 16px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: transparent; /* No background color */
                color: white; /* White font on hover */
                border: 2px solid #4CAF50; /* Keep the green border */
            }
            .stButton>button:active {
                background-color: transparent; /* No background color */
                color: #4CAF50; /* Green font when pressed */
                border: 2px solid #4CAF50; /* Keep the green border */
            }
            </style>
            """, unsafe_allow_html=True)

        if submit_button:
            # Store user's input in a DataFrame
            user_data = {
                'Team':[team_name],
                'Home_Total_Score': [home_total_score],
                'Home_Count': [home_count],
                'Home_Average_Score': [home_total_score / home_count if home_count > 0 else 0],
                'Away_Total_Score': [away_total_score],
                'Away_Count': [away_count],
                'Away_Average_Score': [away_total_score / away_count if away_count > 0 else 0],
                'Number of Goals': [number_of_goals],
                'Sum_First_Goal_Time': [sum_first_goal_time],
                'First_Goal_Count': [first_goal_count],
                'Average_First_Goal_Time': [sum_first_goal_time / first_goal_count if first_goal_count > 0 else 0],
                'True_Neutral_Venue': [true_neutral_venue],
                'False_Neutral_Venue': [false_neutral_venue]
            }

            df_user_input = pd.DataFrame(user_data)
            return df_user_input

user_input_df = get_user_input()

#==============================================================================================================================


###============================================================================================================================

# Paths to saved models
birch_tsne_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/birch_tsne_model.pkl"
scaler_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/standard_scaler.pkl"
combined_df_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/combined_df.csv"
tsne_model_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/tsne_model.pkl"
# label_encoded_path = "C:/Users/Admin/OneDrive/Desktop/ML/ml_Football/label_encoder.pkl"

# Load the KMeans model and the StandardScaler
tsne_file = joblib.load(tsne_model_path)
scaler_file = joblib.load(scaler_path)
birch_file = joblib.load(birch_tsne_path)
# label_encoded_file = joblib.load(label_encoded_path)

# Read the CSV file into a DataFrame
combined_df = pd.read_csv(combined_df_path)

# Assuming 'Team' is the name of the column containing team names
sample_team_names = ["Hong Kong", "Singapore", "Brazil", "Argentina"]

# Filter the DataFrame for the specific teams
sample_filtered_df = combined_df[combined_df['Team'].isin(sample_team_names)]

# Display the filtered DataFrame using Streamlit
st.write(sample_filtered_df)



if user_input_df is not None:
    # Combine the original and user input data
    # st.write(combined_df)
    user_input_combined_df = pd.concat([combined_df, user_input_df], ignore_index=True)
    # st.write(user_input_combined_df)

    label_encoder = LabelEncoder()

    # Apply label encoding (use the existing label encoder if available)
    user_input_combined_df['Team_encoded'] = label_encoder.fit_transform(user_input_combined_df['Team'])
    
    # Standardize the combined dataset using the pre-fitted scaler
    tsne_features = ['Home_Total_Score', 'Home_Count', 'Home_Average_Score',
                     'Away_Total_Score', 'Away_Count', 'Away_Average_Score',
                     'Number of Goals', 'Sum_First_Goal_Time', 'First_Goal_Count',
                     'Average_First_Goal_Time', 'True_Neutral_Venue', 'False_Neutral_Venue']
    
    user_input_combined_df[tsne_features] = scaler_file.transform(user_input_combined_df[tsne_features])
    
    top10_ranking_team = ['Argentina', 'France', 'Spain', 'England', 'Brazil',
                          'Belgium', 'Netherlands', 'Portugal', 'Colombia', 'Italy']

    # Filter the DataFrame for the top 10 teams
    filtered_df = user_input_combined_df[user_input_combined_df['Team'].isin(top10_ranking_team)][['Team', 'Team_encoded']]

    team_encoded_list = filtered_df['Team_encoded'].tolist()

    # Apply t-SNE transformation to the scaled combined dataset
    tsne_result = tsne_file.fit_transform(user_input_combined_df[tsne_features])
    
    # Create a DataFrame for the t-SNE results
    df_tsne = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
    df_tsne['Team_encoded'] = user_input_combined_df['Team_encoded'].values
    df_tsne['Color'] = df_tsne['Team_encoded'].apply(lambda x: 'red' if x in team_encoded_list else 'blue')

    # Visualize t-SNE results
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df_tsne, hue='Color', palette={'red': 'red', 'blue': 'blue'})
    plt.title('Dimensionality Reduction with t-SNE')
    st.pyplot(plt)
    
    # Step 3: Predict the cluster for the t-SNE transformed data
    birch_labels = birch_file.fit_predict(tsne_result)

    # Output the cluster for the user's input
    user_cluster = birch_labels[-1]  # The last entry should correspond to the user input
    st.write(f"User's cluster: {user_cluster}")  # For debugging

    # Define cluster names
    cluster_names = {0: 'Strong Team', 1: 'Weak Team'}
    birch_labels_named = [cluster_names[label] for label in birch_labels]

    # Create a DataFrame for the t-SNE result including BIRCH cluster labels
    df_tsne['Birch_Cluster_TSNE'] = birch_labels_named

    # # DEBUG: Check if user's data is in df_tsne
    # st.write(df_tsne.tail())  # Check if the user's input is included and labeled correctly

    # Ensure the user’s input is in the plot by giving it a unique marker or color
    df_tsne['is_user'] = [False] * (len(df_tsne) - 1) + [True]  # Mark the last row as the user's input

    # Define the custom color palette for clusters
    custom_palette = {
        'Strong Team': '#FFD92F',  # Yellow color for strong teams
        'Weak Team': '#8CE5A1'  # Light green for weak teams
    }

    # Visualize the t-SNE result with clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df_tsne, hue='Birch_Cluster_TSNE',
                    palette=custom_palette, ax=ax, legend='full', s=100)

    # Plot the user’s input as a special marker
    plt.scatter(df_tsne[df_tsne['is_user']]['t-SNE1'], df_tsne[df_tsne['is_user']]['t-SNE2'],
                color='#4284F4', s=200, marker='X', label='User Input')  # Use a red 'X' marker for user's input

    # Compute centroids for each cluster
    centroids = df_tsne.groupby('Birch_Cluster_TSNE')[['t-SNE1', 't-SNE2']].mean().reset_index()

    # Plot centroids
    plt.scatter(x=centroids['t-SNE1'], y=centroids['t-SNE2'], color='red', s=100, marker='X', label='Centroids')

    plt.title('Birch Clustering with t-SNE')
    plt.legend()
    st.pyplot(plt)