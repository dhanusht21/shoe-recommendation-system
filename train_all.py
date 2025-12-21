import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.sparse import csr_matrix

DATA_PATH = os.path.join("data", "ratings.csv")   # change if needed
MODELS_DIR = "modells"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load Data

if os.path.exists(DATA_PATH):
    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Expected columns: user_id, item_id, rating
    if not set(["user_id", "item_id", "rating"]).issubset(df.columns):
        raise ValueError("ratings.csv must contain columns: user_id,item_id,rating")
else:
    # Generate synthetic dataset (20 users, 100 shoes, 20 ratings per user ≈ 400 rows)
    print("ratings.csv not found — generating synthetic dataset for demo.")
    np.random.seed(42)
    num_users = 20
    num_items = 100
    ratings_per_user = 20

    rows = []
    for u in range(1, num_users + 1):
        chosen_items = np.random.choice(range(1, num_items + 1), size=ratings_per_user, replace=False)
        for item in chosen_items:
            rows.append({
                "user_id": f"U{u:03d}",
                "item_id": f"B{item:04d}",
                "rating": int(np.random.randint(1, 6))
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset saved to {DATA_PATH}")

print("Dataset shape:", df.shape)
print(df.head())

# Encoding users and items
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_enc'] = user_encoder.fit_transform(df['user_id'])
df['item_enc'] = item_encoder.fit_transform(df['item_id'])

num_users = df['user_enc'].nunique()
num_items = df['item_enc'].nunique()
print(f"Unique users: {num_users}, unique items: {num_items}")

#Prepare X as user value and y as target to attain
X = df[['user_enc', 'item_enc']].values
y = df['rating'].values

# Optional: Train/test split (not required for final saving, but useful for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train SLP(single hidden layer)
print("\nTraining SLP (MLPRegressor, one hidden layer)...")
slp_model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=42, verbose=True)
slp_model.fit(X_train, y_train)
print("SLP training finished.")

#Train MLP 
print("\nTraining MLP (MLPRegressor, multiple hidden layers)...")
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42, verbose=True)
mlp_model.fit(X_train, y_train)
print("MLP training finished.")

# Building user-item sparse matrix 
print("\nBuilding user-item sparse matrix...")
grouped = df.groupby(['user_enc', 'item_enc'])['rating'].mean().reset_index()
user_item_sparse = csr_matrix(
    (grouped['rating'].values, (grouped['user_enc'].values, grouped['item_enc'].values)),
    shape=(num_users, num_items)
)

#Compute item popularity 
# item_popularity: mean rating per encoded item, sorted descending
item_popularity = df.groupby('item_enc')['rating'].mean().sort_values(ascending=False)

#Save models and artifacts
print("\nSaving models and encoders to 'modells/' ...")
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, 'slp_model.pkl'), 'wb') as f:
    pickle.dump(slp_model, f)

with open(os.path.join(MODELS_DIR, 'mlp_model.pkl'), 'wb') as f:
    pickle.dump(mlp_model, f)

with open(os.path.join(MODELS_DIR, 'user_encoder.pkl'), 'wb') as f:
    pickle.dump(user_encoder, f)

with open(os.path.join(MODELS_DIR, 'item_encoder.pkl'), 'wb') as f:
    pickle.dump(item_encoder, f)

with open(os.path.join(MODELS_DIR, 'user_item_sparse.pkl'), 'wb') as f:
    pickle.dump(user_item_sparse, f)

with open(os.path.join(MODELS_DIR, 'item_popularity.pkl'), 'wb') as f:
    pickle.dump(item_popularity, f)

print("All files saved successfully:")
print(" - modells/slp_model.pkl")
print(" - modells/mlp_model.pkl")
print(" - modells/user_encoder.pkl")
print(" - modells/item_encoder.pkl")
print(" - modells/user_item_sparse.pkl")
print(" - modells/item_popularity.pkl")

#Quick verification
print("\nQuick verification: load one model and run a sample prediction")
with open(os.path.join(MODELS_DIR, 'mlp_model.pkl'), 'rb') as f:
    loaded_mlp = pickle.load(f)

# pick a sample user (encoded) and predict scores for first 5 items
sample_user_enc = 0
items_enc = np.arange(min(5, num_items))
X_sample = np.array([[sample_user_enc, i] for i in items_enc])
preds = loaded_mlp.predict(X_sample)
print("Sample predictions (user_enc=0, items 0..4):", preds)

print("\nTraining script finished. Now restart your Flask app to use the new models.")
