import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score

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
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)

print(df.describe())

X = df[['precedent']]
y = df.prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

sns.pairplot(df, kind='reg')
plt.savefig('previous_result_pairplot.png')

clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=808
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
