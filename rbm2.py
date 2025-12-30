# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 00:21:56 2025

@author: Lenovo
"""

# # Détection de fraude avec RBM
# Idée clé (intuition)

# La fraude est un problème d’anomalie :

# Les transactions normales sont majoritaires

# Les fraudes sont rares et différentes

# On entraîne la RBM uniquement sur des transactions normales

# Les fraudes auront une mauvaise reconstruction

# 👉 Erreur de reconstruction élevée ⇒ transaction suspecte


##  Pipeline général

# Transactions normales
        
# Entraînement RBM
        
# Reconstruction des transactions
        
# Erreur de reconstruction
        
# Score d’anomalie

##  Prétraitement des données
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Chargement
df = pd.read_csv("transactions.csv")

# Supposons que 'is_fraud' = 0 (normal), 1 (fraude)
df_normal = df[df["is_fraud"] == 0]

# Sélection des variables
features = ["amount", "hour", "nb_tx_last_24h"]

# Normalisation [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_normal[features])

X_all = scaler.transform(df[features])

##  RBM pour anomalies (from scratch)
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr

        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b = np.zeros(n_visible)   # biais visibles
        self.c = np.zeros(n_hidden)    # biais cachés

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_h(self, v):
        prob_h = self.sigmoid(np.dot(v, self.W) + self.c)
        h = np.random.binomial(1, prob_h)
        return prob_h, h

    def sample_v(self, h):
        prob_v = self.sigmoid(np.dot(h, self.W.T) + self.b)
        v = np.random.binomial(1, prob_v)
        return prob_v, v

    def train(self, X, epochs=50):
        for epoch in range(epochs):
            for v0 in X:
                # Phase positive
                ph0, h0 = self.sample_h(v0)

                # Phase négative
                pv1, v1 = self.sample_v(h0)
                ph1, h1 = self.sample_h(v1)

                # Mise à jour des paramètres
                self.W += self.lr * (np.outer(v0, ph0) - np.outer(v1, ph1))
                self.b += self.lr * (v0 - v1)
                self.c += self.lr * (ph0 - ph1)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} terminée")

print(dir(rbm))


#  Entraînement & scoring

rbm = RBM(n_visible=X_train.shape[1], n_hidden=10)
rbm.train(X_train, epochs=50)

scores = reconstruction_error(rbm, X_all)
df["anomaly_score"] = scores

#  Seuil de fraude (approche statistique)

## Méthode simple (quantile)

threshold = np.percentile(scores[df["is_fraud"] == 0], 95)
df["fraud_pred"] = (df["anomaly_score"] > threshold).astype(int)

#  Évaluation

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(df["is_fraud"], df["fraud_pred"]))
print("AUC:", roc_auc_score(df["is_fraud"], df["anomaly_score"]))


