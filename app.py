# !pip install flask flask_cors

from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import pipeline
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
try:
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
meal_types = ["soup", "dessert", "meat", "salad", "sandwich", "drink"]
meal_classes = ["breakfast", "lunch", "dinner"]

def classify_meal(user_input):
    meal_result = classifier(user_input, meal_types)
    meal_type = meal_result["labels"][0]

    meal_class = "none"
    for time in meal_classes:
        if time in user_input.lower():
            meal_class = time
            break

    return meal_class, meal_type

import requests

def fetch_data(user_id):
      health_URL = f"https://fitfattt.azurewebsites.net/api/healthInfo/{user_id}"
      diet_URL = f"https://fitfattt.azurewebsites.net/api/dietInfo/{user_id}"
      response1 = requests.get(health_URL)
      response1.raise_for_status()
      data1 = response1.json()
      response2 = requests.get(diet_URL)
      response2.raise_for_status()
      data2 = response2.json()
      return data1["healthInfo"] , data2["dietInfo"]

# !pip install pymongo
import pymongo
from bson.objectid import ObjectId
MONGO_URI = "mongodb+srv://fitfat1:fitfat@fitfat.tdzwb.mongodb.net/"

client = pymongo.MongoClient(MONGO_URI)

db = client["test"]
collection = db["recipes"]
data = list(collection.find())
df = pd.DataFrame(data)
df.drop(columns=['image'], inplace=True)
df.drop(df.loc[df['_id'] == ObjectId("67ded27af1a796cc42250b4e")].index, inplace=True)

col = ["ingredients", "diabetes", "category", "allergy", "diet", "type"]
for i in col:
    df[i] = df[i].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1  else x)
nutrient = ["fat", "protein", "carb", "fiber"]
for i in nutrient:
    df[i] = df[i].str.replace("g", "").astype(float).astype(int)
for column in ["class", "type"]:
    if column in df.columns:
        df[column] = df[column].apply(lambda x: [item.lower() for item in x] if isinstance(x, list) else (x.lower() if isinstance(x, str) else x))
df["category"] = df["category"].apply(lambda x: x.lower())

colum = ["_id", "diet", "diabetes", "allergy", "protein", "calories", "carb", "fat", "type", "class"]
meal_data = df[colum]
# display(meal_data)

meal_data = meal_data.copy()
meal_data["diabetes"] = meal_data["diabetes"].astype(int)

from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(meal_data['class'])

encoders = {}
cat_col = ["diet", "allergy", "type"]
for col in cat_col:
    encoder = LabelEncoder()
    encoder.fit(meal_data[col])
    encoders[col] = encoder
# display(meal_data)

encoded_values_range = {}
for col in cat_col:
    encoded_values_range[f"{col}_encoded"] = set(range(len(encoders[col].classes_)))

def encoding(data):
    if type(data["class"][0]) == list:
      encoded_data = pd.DataFrame(mlb.transform(data['class']), columns=mlb.classes_)
      data_new = data.drop(columns=['class'])
      data_new = pd.concat([data_new, encoded_data], axis=1)
    else:
      encoded_data = pd.DataFrame(0, index=data.index, columns=mlb.classes_)
      for i, val in enumerate(data["class"]):
          if val in mlb.classes_:
              encoded_data.loc[i, val] = 1
      data_new = data.drop(columns=['class'])
      data_new = pd.concat([data_new, encoded_data], axis=1)
    for col in cat_col:
      cat = f"{col}_encoded"
      data_new[cat] = data_new[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
      data_new = data_new.drop(columns=[col])
    return data_new

data_new = encoding(meal_data)

for col in cat_col:
    print(f"\nColumn: {col}")
    print("Original values:", encoders[col].classes_)
    print("Encoded values:")
    for i, val in enumerate(encoders[col].classes_):
        print(f"  {val} → {i}")

numerical_features = ['protein', 'calories', 'carb', 'fat']
def scale_numerical_features(meals_df, numerical_features):
  scaler = StandardScaler()
  meals_df[numerical_features] = scaler.fit_transform(meals_df[numerical_features])
  return meals_df , scaler
data_new, scaler = scale_numerical_features(data_new, numerical_features)
# display(data_new)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

base_columns = ['diet_encoded', 'allergy_encoded', 'diabetes', 'type_encoded', 'breakfast', 'dinner', 'lunch']
X_train, X_temp = train_test_split(data_new[base_columns], test_size=0.2, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(train_data, val_data, input_dim):

    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 90
    patience = 10
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_data)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_data)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    model.eval()
    return model

base_model = train_autoencoder(X_train, X_val, X_train.shape[1])

models_cache = {
    tuple(base_columns): base_model
}

def get_or_train_model(columns):
    columns_tuple = tuple(sorted(columns))

    if columns_tuple in models_cache:
        return models_cache[columns_tuple]

    print(f"train new model for columns: {columns}")

    X_train_new, X_temp_new = train_test_split(data_new[columns], test_size=0.2, random_state=42)
    X_val_new, X_test_new = train_test_split(X_temp_new, test_size=0.5, random_state=42)

    X_train_tensor = torch.tensor(X_train_new.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_new.values, dtype=torch.float32)

    new_model = train_autoencoder(X_train_tensor, X_val_tensor, len(columns))

    models_cache[columns_tuple] = new_model

    return new_model

def get_valid_columns_for_user(user_new, base_columns):
    valid_columns = []

    for col in base_columns:
        if col in ['diabetes', 'breakfast', 'dinner', 'lunch']:
            if user_new[col].iloc[0] != 0:
                valid_columns.append(col)
        elif col == 'type_encoded':
            user_value = user_new[col].iloc[0]
            if user_value != -1 and user_value in encoded_values_range[col]:
                valid_columns.append(col)
        elif col in ['diet_encoded', 'allergy_encoded']:
            user_value = user_new[col].iloc[0]
            if user_value != -1 and user_value in encoded_values_range[col]:
                valid_columns.append(col)
                print(f"Included column {col} - value: {user_value}")
            else:
                print(f"Excluded column {col} - invalid value: {user_value}")

    return valid_columns

def filter_meals_by_allergy(meal_data, user_allergy_encoded):
    safe_meals_mask = meal_data['allergy_encoded'] == user_allergy_encoded
    return meal_data[safe_meals_mask].copy()

def get_recommendations_with_allergy_priority(user_new, valid_columns, data_new, meal_data, top_k=5):

    has_allergy = 'allergy_encoded' in valid_columns
    if has_allergy:
        user_allergy_encoded = user_new['allergy_encoded'].iloc[0]
        safe_meal_indices = data_new['allergy_encoded'] == user_allergy_encoded
        safe_data = data_new[safe_meal_indices].copy()
        safe_meal_data = meal_data[safe_meal_indices].copy()

        if len(safe_data) > 0:
            current_model = get_or_train_model(valid_columns)
            X_safe = torch.tensor(safe_data[valid_columns].values, dtype=torch.float32)
            with torch.no_grad():
                safe_meal_embeddings = current_model.encoder(X_safe).numpy()
                user_embedding = current_model.encoder(
                    torch.tensor(user_new[valid_columns].values, dtype=torch.float32)
                ).numpy()

            similarities = cosine_similarity(user_embedding, safe_meal_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_meal_id = str(safe_meal_data.iloc[best_idx]["_id"])
            return [{"meal_id": best_meal_id, "is_allergy_safe": True}]

    current_model = get_or_train_model(valid_columns)
    X_all = torch.tensor(data_new[valid_columns].values, dtype=torch.float32)
    with torch.no_grad():
        meal_embeddings = current_model.encoder(X_all).numpy()
        user_embedding = current_model.encoder(
            torch.tensor(user_new[valid_columns].values, dtype=torch.float32)
        ).numpy()

    similarities = cosine_similarity(user_embedding, meal_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_meal_id = str(meal_data.iloc[best_idx]["_id"])
    is_safe = not has_allergy or data_new.iloc[best_idx]["allergy_encoded"] == user_new["allergy_encoded"].iloc[0]

    return [{"meal_id": best_meal_id, "is_allergy_safe": is_safe}]

@app.route("/recommend", methods=["POST"])

def recommend_meal():
  data = request.get_json()
  user_input = data.get("query", "")
  user_id = data.get("userId")
  meal_class , meal_type = classify_meal(user_input)
  health_data, diet_data = fetch_data(user_id)
  merged_data = {
              "userId": user_id,
              "allergy": health_data.get("foodAllergies", "").lower(),
              "diabetes": int(health_data.get("diabetes", False)),
              "diet": diet_data.get("dietType", "").lower(),
              "protein": diet_data.get("macronutrientGoals", {}).get("proteins"),
              "carb": diet_data.get("macronutrientGoals", {}).get("carbs"),
              "fat": diet_data.get("macronutrientGoals", {}).get("fats"),
              "calories": diet_data.get("macronutrientGoals", {}).get("calories"),
              "type": meal_type,
              "class": meal_class
          }

  user_df = pd.DataFrame([merged_data])
  base_columns = ['diet_encoded', 'allergy_encoded', 'diabetes', 'type_encoded', 'breakfast', 'dinner', 'lunch']
  numerical_features = ['protein', 'calories', 'carb', 'fat']
  user_new = encoding(user_df)
  user_new[numerical_features] = scaler.transform(user_new[numerical_features])
  valid_columns = get_valid_columns_for_user(user_new, base_columns)

  if not valid_columns:
      return jsonify({"error": "No suitable recommendation "})
  recommendations = get_recommendations_with_allergy_priority(
          user_new, valid_columns, data_new, meal_data, top_k=5
  )

  if recommendations:
      best_recommendation = recommendations[0]

      response = {"recommended_meal_id": best_recommendation["meal_id"]}
      if not best_recommendation["is_allergy_safe"]:
          response["warning"] = "The recommended meal may not be suitable for your allergy. Please check ingredients carefully."

      return jsonify(response)
  else:
      return jsonify({"error": "No suitable recommendations found"})

# !pip install pyngrok
# from pyngrok import ngrok

# ngrok.set_auth_token("2xRwTGEb3FujRu2pPJHHMU7JUG9_4LKkn5T7TQmmGNvTpWbNJ")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # public_url = ngrok.connect(port)
    # print(" * ngrok tunnel:", public_url)
    app.run(host="0.0.0.0", port=port)
