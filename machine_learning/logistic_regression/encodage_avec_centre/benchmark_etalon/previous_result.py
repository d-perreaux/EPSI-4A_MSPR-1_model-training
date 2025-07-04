import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

from db_query.query import DatabaseQuery


# exemple de requête
# sql = "SELECT * FROM users WHERE id = %s"
# params = (1,)  # Tuple avec un seul élément
#
# cursor.execute(sql, params)

# autre exemple
# users = DatabaseQuery.execute("SELECT * FROM users WHERE age > %s", (25,), fetch_all=True)

query = """
    SELECT encodage_avec_centre_gagnant as prediction, encodage_avec_centre_gagnant_precedent as precedent FROM legislative_per_cir
    WHERE encodage_avec_centre_gagnant_precedent IS NOT NULL
    AND encodage_avec_centre_gagnant IS NOT NULL
    AND annee IN (2017, 2022, 2024)
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)

print(df.describe())

X = df[['precedent']]
y = df.prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

sns.pairplot(df, kind='reg')
plt.savefig('previous_result_pairplot.png')

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Matrice de confusion')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("Precision (pondérée) :", precision_score(y_test, y_pred, average='weighted', zero_division=1))
print("Recall (pondéré) :", recall_score(y_test, y_pred, average='weighted', zero_division=1))
print("F1-score (pondéré) :", f1_score(y_test, y_pred, average='weighted', zero_division=1))

y_proba = clf.predict_proba(X_test)

print("ROC AUC (ovr) :", roc_auc_score(y_test, y_proba, multi_class='ovr'))

print("Total data:", len(df))
print("Train size:", len(X_train))
print("Test size:", len(X_test))