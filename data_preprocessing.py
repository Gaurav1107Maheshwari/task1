import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Load the CSV file
file_path = "1. Regression - Module - (Housing Prices).csv"
df = pd.read_csv(file_path)

# Display basic info
print("Initial Data Shape:", df.shape)
print(df.info())
print(df.head())

# Handling Missing Values
# 1. For numerical columns, fill missing values with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 2. For categorical columns, fill missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("Null values after imputation:\n", df.isnull().sum())

# Encoding Categorical Variables
# 1. Label Encoding for ordinal features or binary categories
label_encodable = [col for col in cat_cols if df[col].nunique() == 2]
le = LabelEncoder()
for col in label_encodable:
    df[col] = le.fit_transform(df[col])

# 2. One-hot encoding for nominal features
onehot_encodable = [col for col in cat_cols if df[col].nunique() > 2]
df = pd.get_dummies(df, columns=onehot_encodable, drop_first=True)

print("Data shape after encoding:", df.shape)

# Feature Scaling
# You can choose between StandardScaler or MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler() # Uncomment if you prefer MinMax scaling

# Exclude target variable from scaling
target_col = 'SalePrice'  # Change this if your target column has a different name
features = df.drop(columns=[target_col])
target = df[target_col]

features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Combine scaled features with target
df_final = pd.concat([features_scaled_df, target.reset_index(drop=True)], axis=1)

print("Final processed data shape:", df_final.shape)
print(df_final.head())

# Optionally, split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled_df, target, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)