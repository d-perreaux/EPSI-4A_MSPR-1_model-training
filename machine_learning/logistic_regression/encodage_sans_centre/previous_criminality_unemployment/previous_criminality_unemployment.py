import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve

from db_query.query import DatabaseQuery


query = """
    SELECT 
    encodage_sans_centre_gagnant as prediction, encodage_sans_centre_gagnant_precedent as precedent, criminality_per_cir.code_circonscription as code_circonscirption,
    taux_pour_mille, taux_chomage  
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    INNER JOIN unemployment_per_cir ON legislative_per_cir.annee = unemployment_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = unemployment_per_cir.code_circonscription
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)

print(df.describe())

X = df[['precedent', 'taux_pour_mille', 'taux_chomage']]
y = df.prediction
labels = [1, 2, 3, 4]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

print(X_train)
print(X_test)
print(y_train.value_counts())
print(y_test.value_counts())

plt.figure()
sns.pairplot(df, kind='reg')
plt.savefig('previous_result_criminality_unemployment_pairplot.png')

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
plt.figure()
display.plot()
plt.savefig('confusion_matrix.png')

y_hat_proba = clf.predict_proba(X_test)

print(y_hat_proba)

plt.figure()
sns.histplot(y_hat_proba[:, 0], kde=True, label="Classe 1")  # Classe 1
plt.legend()
plt.savefig(f'histplot.predict_proba_1.png')

plt.figure()
sns.histplot(y_hat_proba[:, 1], kde=True, label="Classe 2")  # Classe 1
plt.legend()
plt.savefig(f'histplot.predict_proba_2.png')

plt.figure()
sns.histplot(y_hat_proba[:, 2], kde=True, label="Classe 3")  # Classe 1
plt.legend()
plt.savefig(f'histplot.predict_proba_3.png')

plt.figure()
sns.histplot(y_hat_proba[:, 3], kde=True, label="Classe 4")  # Classe 1
plt.legend()
plt.savefig(f'histplot.predict_proba_4.png')

print(np.min(y_hat_proba[:, 0]), np.max(y_hat_proba[:, 0]))
print(np.min(y_hat_proba[:, 1]), np.max(y_hat_proba[:, 1]))
print(np.min(y_hat_proba[:, 2]), np.max(y_hat_proba[:, 2]))
print(np.min(y_hat_proba[:, 3]), np.max(y_hat_proba[:, 3]))
