import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

from db_query.query import DatabaseQuery


query = """
    SELECT encodage_sans_centre_gagnant as prediction, encodage_sans_centre_gagnant_precedent as precedent FROM legislative_per_cir
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
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
