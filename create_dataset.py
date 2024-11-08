import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data
data = pd.read_csv("data/02-14-2018.csv")

replace_dict = {np.nan: 0, " ": 0}
columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
       'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

# Replace specified values
for cols in columns:
    data[cols] = data[cols].replace(replace_dict)

# Replace infinities with NaN, then handle NaN as specified
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

# Reset index and drop duplicates
data = data.reset_index(drop=True)
data = data.drop_duplicates()

# Drop specified columns
data.drop(columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
                   'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Bwd PSH Flags', 'Fwd URG Flags',
                   'Bwd URG Flags'], axis=1, inplace=True)

# Preprocessing
categorical_features = ["Protocol"]
binary_features = ["Label"]
continuous_features = data.columns.drop(categorical_features + binary_features)

# Creating a transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), continuous_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)

# Fit and transform the data
data_processed = preprocessor.fit_transform(data)

# Convert the NumPy array to a DataFrame
data_processed = pd.DataFrame(data_processed, columns=preprocessor.get_feature_names_out())

# Convert binary features to torch tensors
for feature in binary_features:
    data_processed.loc[:, feature] = data[feature].astype(int).values

data_processed.to_parquet("data/cic_processed.parquet")

# Create the mask for features excluding the label
mask = np.zeros(data_processed.shape[1] - 1, dtype=int)

data_columns_without_label = data_processed.columns.drop("Label")

# Set mask values for continuous features
num_feature_names = preprocessor.named_transformers_["num"].get_feature_names_out(continuous_features)
num_indices = [data_columns_without_label.get_loc("num__" + name) for name in num_feature_names]
mask[num_indices] = 1

# Set mask values for binary features
# binary_indices = [data_columns_without_label.get_loc(col) for col in binary_features if col != "Label"]
# mask[binary_indices] = 1

print("Mask for features:", mask)
torch.save(torch.tensor(mask), "data/mask.pt")
