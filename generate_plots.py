import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Style graphique
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

print("Génération des schémas et graphiques en cours...")

# 1. Chargement des données
train_df = pd.read_csv('reservations_train.csv')
train_df['date_reservation'] = pd.to_datetime(train_df['date_reservation'])
train_df['date_arrivee'] = pd.to_datetime(train_df['date_arrivee'])
train_df = train_df.sort_values('date_reservation').reset_index(drop=True)

# Feature engineering rapide
train_df['delai_anticipation'] = (train_df['date_arrivee'] - train_df['date_reservation']).dt.days
train_df['mois_reservation'] = train_df['date_reservation'].dt.month

# --- GRAPHIQUE 1 : Analyse Exploratoire (EDA) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution de la cible
sns.countplot(data=train_df, x='reservation_annulee', palette='viridis', ax=axes[0])
axes[0].set_title('Distribution de la Cible (0: Maintenue, 1: Annulée)')
axes[0].set_xlabel('Statut de Réservation')
axes[0].set_ylabel('Nombre de Réservations')

# Impact du délai d'anticipation sur l'annulation
sns.boxplot(data=train_df, x='reservation_annulee', y='delai_anticipation', palette='Set2', ax=axes[1])
axes[1].set_title("Délai d'anticipation vs Annulation")
axes[1].set_xlabel('Statut (0: Maintenue, 1: Annulée)')
axes[1].set_ylabel("Délai d'anticipation (jours)")

plt.tight_layout()
plt.savefig('1_eda_distributions.png', dpi=300)
plt.close()
print("✓ Graphique 1 sauvegardé : 1_eda_distributions.png")

# --- Entraînement Modèle pour Visuels ---
X = train_df.drop(columns=['reservation_id', 'date_reservation', 'date_arrivee', 'reservation_annulee'])
y = train_df['reservation_annulee']

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, verbose=-1))
])

# Split temporel pour la matrice de confusion
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

model.fit(X_tr, y_tr)
val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.22).astype(int)

# --- GRAPHIQUE 2 : Matrice de Confusion ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_val, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Maintenue (0)', 'Annulée (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matrice de Confusion (Seuil = 0.22)')
plt.grid(False)
plt.tight_layout()
plt.savefig('2_matrice_confusion.png', dpi=300)
plt.close()
print("✓ Graphique 2 sauvegardé : 2_matrice_confusion.png")

# --- GRAPHIQUE 3 : Importance des Variables ---
model.fit(X, y)
clf = model.named_steps['classifier']
cat_encoded = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
all_features = num_features + list(cat_encoded)

importances = pd.Series(clf.feature_importances_, index=all_features).sort_values(ascending=False).head(12)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='magma')
plt.title('Top 12 des Variables les plus Importantes (LightGBM)')
plt.xlabel("Score d'Importance (Gain / Split)")
plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300)
plt.close()
print("✓ Graphique 3 sauvegardé : 3_feature_importance.png")

print("\nTous les visuels ont été générés avec succès !")