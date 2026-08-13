import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 50)
print("1. CHARGEMENT ET PRÉPARATION DES DONNÉES")
print("=" * 50)

train_df = pd.read_csv('reservations_train.csv')
test_df = pd.read_csv('reservations_test.csv')

# Conversion des dates
for col in ['date_reservation', 'date_arrivee']:
    train_df[col] = pd.to_datetime(train_df[col])
    test_df[col] = pd.to_datetime(test_df[col])

# Tri chronologique sur le train
train_df = train_df.sort_values('date_reservation').reset_index(drop=True)


# ---------------------------------------------------------
# ÉTAPE 4 : FEATURE ENGINEERING (Sans fuite de données)
# ---------------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    # 1. Variables temporelles
    df['delai_anticipation'] = (df['date_arrivee'] - df['date_reservation']).dt.days
    df['mois_reservation'] = df['date_reservation'].dt.month
    df['jour_semaine_reservation'] = df['date_reservation'].dt.dayofweek
    df['mois_arrivee'] = df['date_arrivee'].dt.month
    df['jour_semaine_arrivee'] = df['date_arrivee'].dt.dayofweek

    # 2. Composition du séjour
    if 'adultes' in df.columns and 'enfants' in df.columns:
        df['total_personnes'] = df['adultes'] + df['enfants']

    # 3. Historique client (ratio d'annulation passé)
    if 'annulations_passees' in df.columns and 'reservations_passees' in df.columns:
        df['ratio_annulations_passees'] = df['annulations_passees'] / (df['reservations_passees'] + 1)

    # Suppression des colonnes non prédictives / identifiants / dates brutes
    cols_to_drop = ['reservation_id', 'date_reservation', 'date_arrivee']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


train_fe = feature_engineering(train_df)
test_fe = feature_engineering(test_df)

X = train_fe.drop(columns=['reservation_annulee'])
y = train_fe['reservation_annulee']
X_test = test_fe.copy()

# Identification des colonnes
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ---------------------------------------------------------
# PRÉTRAITEMENT (Pipeline Scikit-Learn)
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# ---------------------------------------------------------
# ÉTAPE 2 : BASELINE (RÉGRESSION LOGISTIQUE)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("2. ÉVALUATION BASELINE (RÉGRESSION LOGISTIQUE)")
print("=" * 50)

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=SEED, max_iter=1000))
])

# Validation Temporelle (TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)
f1_baseline_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    baseline_pipeline.fit(X_tr, y_tr)
    y_pred = baseline_pipeline.predict(X_val)
    score = f1_score(y_val, y_pred, pos_label=1)
    f1_baseline_scores.append(score)

print(f"F1-score moyen Baseline (Seuil 0.5) : {np.mean(f1_baseline_scores):.4f}")

# ---------------------------------------------------------
# ÉTAPE 3 : MODÈLE AVANCÉ (LightGBM) & OPTIMISATION DU SEUIL
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("3. ÉVALUATION LIGHTGBM & OPTIMISATION DU SEUIL")
print("=" * 50)

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=SEED, n_estimators=150, learning_rate=0.05, verbose=-1))
])

# Entraînement et Recherche de Seuil sur le dernier fold temporel
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

lgbm_pipeline.fit(X_tr, y_tr)
val_probs = lgbm_pipeline.predict_proba(X_val)[:, 1]

# Recherche du seuil optimal pour maximiser le F1-Score sur la classe 1
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.02):
    preds = (val_probs >= threshold).astype(int)
    score = f1_score(y_val, preds, pos_label=1)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"Seuil Optimal Trouvé : {best_threshold:.2f}")
print(f"F1-score LightGBM Validé : {best_f1:.4f}")

# ---------------------------------------------------------
# ÉTAPE 6 : SOUMISSION (Génération de submission.csv)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("4. GÉNÉRATION DE SUBMISSION.CSV")
print("=" * 50)

# Entraînement final du modèle sur la totalité des données Train
lgbm_pipeline.fit(X, y)

# Prédictions sur le jeu Test
test_probs = lgbm_pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

# Création du DataFrame final
submission = pd.DataFrame({
    "reservation_id": test_df["reservation_id"],
    "probabilite_annulation": test_probs,
    "reservation_annulee": test_preds
})

submission.to_csv("submission.csv", index=False)

print("Fichier 'submission.csv' généré avec succès !")
print(f"Nombre de lignes : {len(submission)} (Attendu : 2000)")
print("Aperçu des premières lignes :")
print(submission.head())