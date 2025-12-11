from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---- Load models and data ----
with open('modells/user_item_sparse.pkl', 'rb') as f:
    user_item_sparse = pickle.load(f)
user_item = pd.DataFrame.sparse.from_spmatrix(user_item_sparse)
user_item.index.name = 'user'
user_item.columns.name = 'item'

with open('modells/item_popularity.pkl', 'rb') as f:
    item_popularity = pickle.load(f)

with open('modells/user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('modells/item_encoder.pkl', 'rb') as f:
    item_encoder = pickle.load(f)

with open('modells/slp_model.pkl', 'rb') as f:
    slp_model = pickle.load(f)

with open('modells/mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

# ---- Load full product catalog ----
products_df = pd.read_csv('data/ratings.csv')  # item_id,title,image_url
product_catalog = products_df.set_index('item_id').to_dict('index')


# ---- Collaborative Filtering Recommender ----
def recommend_collab(user_id, top_n=6):
    # If user is unknown, show top popular items
    if user_id not in user_encoder.classes_:
        top_items_enc = item_popularity.index[:top_n].tolist()
    else:
        u_enc = user_encoder.transform([user_id])[0]
        if u_enc not in user_item.index:
            top_items_enc = item_popularity.index[:top_n].tolist()
        else:
            user_ratings = user_item.loc[u_enc]
            high_rated = user_ratings[user_ratings > 0].sort_values(ascending=False).index.tolist()
            top_items_enc = high_rated[:top_n]

            # Fill remaining with popular items if not enough
            if len(top_items_enc) < top_n:
                popular_items = [item for item in item_popularity.index if item not in top_items_enc]
                top_items_enc.extend(popular_items[:top_n - len(top_items_enc)])

    # Decode all items at once
    top_items_dec = item_encoder.inverse_transform(top_items_enc)

    # Map to product catalog
    top_items = [
        product_catalog.get(item_id, {'title': str(item_id), 'image_url': '/static/images/default.jpg'})
        for item_id in top_items_dec
    ]

    return top_items



# ---- Neural Network Recommender (SLP & MLP) ----
def recommend_nn(user_id, model, top_n=8):

    # If unknown user → show popular items
    if user_id not in user_encoder.classes_:
        top_items_enc = item_popularity.head(top_n).index.tolist()
        return [
            product_catalog.get(
                item_encoder.inverse_transform([i])[0],
                {"title": str(i), "image_url": "/static/images/default.jpg"}
            )
            for i in top_items_enc
        ]

    # Encode user
    user_enc = user_encoder.transform([user_id])[0]

    scores = []

    # Predict score for all items
    for item_enc in range(len(item_encoder.classes_)):
        X = np.array([[user_enc, item_enc]])
        score = model.predict(X)[0]   # <-- NOW the model is used
        scores.append((item_enc, score))

    # Sort by predicted score (descending)
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    # Select top N
    top_items_enc = [item for item, score in scores_sorted[:top_n]]

    # Decode to real IDs
    return [
        product_catalog.get(
            item_encoder.inverse_transform([i])[0],
            {"title": str(i), "image_url": "/static/images/default.jpg"}
        )
        for i in top_items_enc
    ]

# ---- Flask route ----
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_id = None
    method = 'collab'

    if request.method == 'POST':
        raw_user = request.form.get('user_id')
        method = request.form.get('method', 'collab')
        user_id = raw_user.strip()

        if method == 'collab':
            recommendations = recommend_collab(user_id)
        elif method == 'slp':
            recommendations = recommend_nn(user_id, slp_model,top_n=5)
        elif method == 'mlp':
            recommendations = recommend_nn(user_id, mlp_model,top_n=8)

    return render_template('index.html', recommendations=recommendations, user_id=user_id, method=method)


if __name__ == '__main__':
    app.run(debug=True)
