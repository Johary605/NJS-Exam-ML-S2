import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cellule 1 : Titre & Introduction (Markdown)
c1 = nbf.v4.new_markdown_cell("""# 🏨 Hackathon ML & Data Science — Atlantic Haven Hotels
**Objectif :** Prédire l'annulation des réservations (`reservation_annulee`) et optimiser le F1-score sur la classe 1.
""")

# Cellule 2 : Imports et Configuration (Code)
c2 = nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
print("Bibliothèques chargées avec succès !")
""")

# Cellule 3 : Chargement des Données (Code)
c3 = nbf.v4.new_code_cell("""# 1. Chargement et tri temporel
train_df = pd.read_csv('reservations_train.csv')
test_df = pd.read_csv('reservations_test.csv')

for col in ['date_reservation', 'date_arrivee']:
    train_df[col] = pd.to_datetime(train_df[col])
    test_df[col] = pd.to_datetime(test_df[col])

train_df = train_df.sort_values('date_reservation').reset_index(drop=True)

print(f"Dimensions Train : {train_df.shape}")
print(f"Dimensions Test  : {test_df.shape}")
""")

# Cellule 4 : Feature Engineering (Code)
c4 = nbf.v4.new_code_cell("""# 2. Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df['delai_anticipation'] = (df['date_arrivee'] - df['date_reservation']).dt.days
    df['mois_reservation'] = df['date_reservation'].dt.month
    df['jour_semaine_reservation'] = df['date_reservation'].dt.dayofweek
    df['mois_arrivee'] = df['date_arrivee'].dt.month
    df['jour_semaine_arrivee'] = df['date_arrivee'].dt.dayofweek

    if 'adultes' in df.columns and 'enfants' in df.columns:
        df['total_personnes'] = df['adultes'] + df['enfants']

    if 'annulations_passees' in df.columns and 'reservations_passees' in df.columns:
        df['ratio_annulations_passees'] = df['annulations_passees'] / (df['reservations_passees'] + 1)

    cols_to_drop = ['reservation_id', 'date_reservation', 'date_arrivee']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

train_fe = feature_engineering(train_df)
test_fe = feature_engineering(test_df)

X = train_fe.drop(columns=['reservation_annulee'])
y = train_fe['reservation_annulee']
X_test = test_fe.copy()

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])
""")

# Cellule 5 : Modélisation et Soumission (Code)
c5 = nbf.v4.new_code_cell("""# 3. Entraînement LightGBM et Génération de submission.csv
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=SEED, n_estimators=150, learning_rate=0.05, verbose=-1))
])

# Entraînement sur l'ensemble du train
lgbm_pipeline.fit(X, y)

# Prédictions sur le test avec le seuil optimal (0.22)
test_probs = lgbm_pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.22).astype(int)

submission = pd.DataFrame({
    "reservation_id": test_df["reservation_id"],
    "probabilite_annulation": test_probs,
    "reservation_annulee": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Fichier submission.csv généré avec succès !")
print(submission.head())
""")

nb['cells'] = [c1, c2, c3, c4, c5]

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Fichier notebook.ipynb généré et rempli avec succès !")