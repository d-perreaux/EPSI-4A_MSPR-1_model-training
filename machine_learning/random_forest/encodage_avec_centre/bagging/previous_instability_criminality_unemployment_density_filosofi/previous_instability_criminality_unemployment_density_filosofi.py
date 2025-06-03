import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize

from db_query.query import DatabaseQuery


query = """
    SELECT 
    encodage_avec_centre_gagnant as prediction, encodage_avec_centre_gagnant_precedent as precedent, criminality_per_cir.code_circonscription as code_circonscirption,
    taux_pour_mille, taux_chomage, density, instabilite_avec_centre, poids_nuance_avec_centre, desir_changement_avec_centre, is_2024, 
    ratio_menages_proprietaires, ratio_menages_en_maisons, ratio_menages_en_logements_sociaux, ratio_menages_pauvres, ratio_individus_18_24, ratio_individus_65_79  
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    INNER JOIN unemployment_per_cir ON legislative_per_cir.annee = unemployment_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = unemployment_per_cir.code_circonscription
    INNER JOIN density_population ON legislative_per_cir.code_du_departement = density_population.code_departement
    INNER JOIN statistiques_circonscriptions ON legislative_per_cir.code_de_la_circonscription = statistiques_circonscriptions.code_de_la_circonscription AND legislative_per_cir.annee = statistiques_circonscriptions.annee_legislative
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)

print(df.describe())

X = df[['precedent', 'taux_pour_mille', 'taux_chomage', 'density',
        'instabilite_avec_centre', 'poids_nuance_avec_centre', 'desir_changement_avec_centre','ratio_menages_proprietaires',
        'ratio_menages_en_maisons', 'ratio_menages_en_logements_sociaux', 'is_2024', 'ratio_menages_pauvres', 'ratio_individus_18_24', 'ratio_individus_65_79']]
y = df.prediction
labels = [1, 2, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)
print(y_train.value_counts())
print(y_test.value_counts())

plt.figure()
sns.pairplot(df, kind='reg')
plt.savefig('previous_result_criminality_unemployment_pairplot.png')

tree_counts = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
accuracy = []
for n_estimator in tree_counts:
    clf = RandomForestClassifier(
        n_estimators=n_estimator,
        max_depth=2,
        max_features=3,
        random_state=8
    )
    clf.fit(X_train, y_train)
    accuracy.append(clf.score(X_test, y_test))

    print(
        f"{n_estimator} trees \t accuracy test: {np.round(clf.score(X_test, y_test), 3)} \t accuracy train {np.round(clf.score(X_train, y_train), 3)}", )

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1, 1)
plt.plot(tree_counts, accuracy)
plt.plot(tree_counts, accuracy,'*')
ax.grid(True, which = 'both')
ax.set_title('Accuracy on test vs n_estimators')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))
# plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('accuracy_on_tests_vs_nèestimators')

# Réduction de la varianec
# Chaque modèle d'arbre n'a plus de contrainte de profondeur

tree_counts = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150]

accuracy = []

for n_estimator in tree_counts:
    clf = RandomForestClassifier(
        n_estimators=n_estimator,
        max_depth = None,
        max_features = None,
        random_state = 8
        )

    clf.fit(X_train, y_train)
    accuracy.append({
        'n': n_estimator,
        'test': clf.score(X_test, y_test),
        'train': clf.score(X_train, y_train),
    })

    print(f"{n_estimator} trees \t accuracy test: {np.round(clf.score(X_test, y_test), 3)} \t accuracy train {np.round(clf.score(X_train, y_train), 3)}", )


accuracy = pd.DataFrame(accuracy)
accuracy['delta'] = np.abs(accuracy.train - accuracy.test)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(accuracy.n, accuracy.train, label = 'score train')
plt.plot(accuracy.n, accuracy.train,'*')

plt.plot(accuracy.n, accuracy.test, label = 'score test')
plt.plot(accuracy.n, accuracy.test,'*')

# plt.plot(accuracy.n, accuracy.delta, label = 'delta')
# plt.plot(accuracy.n, accuracy.delta,'*')

ax.grid(True, which = 'both')
ax.set_title('Accuracy')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Accuracy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

ax.legend()
# --
ax = fig.add_subplot(1, 2, 2)
plt.plot(accuracy.n, accuracy.delta, label = 'delta')
plt.plot(accuracy.n, accuracy.delta,'*')

ax.grid(True, which='both')
ax.set_title('Différence score(test) - score(train) ')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Différence score(test) - score(train)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

ax.legend()
plt.tight_layout()
fig.savefig('accuracy_and_delta_sore_vs_n_estimator(no_max_depth).png')


# 90 arbres, sans max_depth ni max_features
clf = RandomForestClassifier(
    n_estimators=90,
    max_depth=None,
    max_features=None,
    random_state=8
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Matrice de confusion :")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Matrice de confusion (90 arbres)")
plt.tight_layout()
plt.savefig("confusion_matrix_90_estimators.png")

# Précision macro (pour plusieurs classes)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Précision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")

# ROC AUC pour classes multiples
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=labels)
y_score = clf.predict_proba(X_test)

# On vérifie que predict_proba est bien sous forme [n_samples, n_classes]
roc_auc = roc_auc_score(y_test_binarized, y_score, average="macro", multi_class="ovr")
print(f"ROC AUC (macro, OVR): {roc_auc:.3f}")

importances = clf.feature_importances_
feature_names = X.columns  # ou liste manuelle si nécessaire

# Création d'un DataFrame pour trier et afficher joliment
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importances_df)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances_df, palette='viridis')
plt.title('Importance des variables (Random Forest - Bagging)')
plt.tight_layout()
plt.savefig('feature_importances_bagging.png')

