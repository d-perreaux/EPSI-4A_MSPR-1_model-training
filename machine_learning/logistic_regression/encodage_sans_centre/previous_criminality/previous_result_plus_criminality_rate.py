import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

from db_query.query import DatabaseQuery


query = """
    SELECT 
    encodage_sans_centre_gagnant as prediction, encodage_sans_centre_gagnant_precedent as precedent, code_circonscription,
    taux_pour_mille  
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)

print(df.describe())

X = df[['precedent', 'taux_pour_mille']]
y = df.prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

plt.figure()
sns.pairplot(df, kind='reg')
plt.savefig('previous_result_plus_criminality_pairplot.png')

plt.figure()
sns.scatterplot(x='code_circonscription', y='taux_pour_mille', data=df, hue='prediction')
plt.savefig('criminality_scatterplot.png')


clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
