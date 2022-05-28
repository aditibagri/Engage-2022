import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px

#define URL where dataset is located


#read in data
data_import = pd.read_csv(r"C:\Users\pc\Desktop\engage_final\engage_final.csv")


numerical_columns_list = []
categorical_columns_list = []

for i in data_import.columns:
        if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
            numerical_columns_list.append(data_import[i])
        else:
            categorical_columns_list.append(data_import[i])

numerical_data = pd.concat(numerical_columns_list, axis=1)
categorical_data = pd.concat(categorical_columns_list, axis=1)

numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

scaler = StandardScaler()
scaled_values = scaler.fit_transform(numerical_data)
pca = PCA()
pca_data = pca.fit_transform(scaled_values)
pca_data = pd.DataFrame(pca_data)

PC_values = np.arange(pca.n_components_) + 1

plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()













