import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from db_query.query import DatabaseQuery


query = """
    SELECT 
    encodage_avec_centre_gagnant as prediction, encodage_avec_centre_gagnant_precedent as precedent, criminality_per_cir.code_circonscription as code_circonscirption,
    taux_pour_mille, taux_chomage, density, instabilite_avec_centre, poids_nuance_avec_centre, desir_changement_avec_centre,
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
        'ratio_menages_en_maisons', 'ratio_menages_en_logements_sociaux', 'ratio_menages_pauvres', 'ratio_individus_18_24', 'ratio_individus_65_79']]
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

# TODO
# REGARDER CE QUE FAIT CLASS_WIEGHT
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
plt.figure()
display.plot()
plt.savefig('confusion_matrix.png')

y_hat_proba = clf.predict_proba(X_test_scaled)


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


print(np.min(y_hat_proba[:, 0]), np.max(y_hat_proba[:, 0]))
print(np.min(y_hat_proba[:, 1]), np.max(y_hat_proba[:, 1]))
print(np.min(y_hat_proba[:, 2]), np.max(y_hat_proba[:, 2]))

# Precision, Recall and ROC_AUC
print(f"precision : {precision_score(y_test, y_pred, average='weighted', zero_division=1)}")
print(f"recall : {recall_score(y_test, y_pred, average='weighted',zero_division=1)}")
print(f"roc auc : {roc_auc_score(y_test, y_hat_proba, multi_class='ovr')}")

# Nombre de classes
n_classes = len(labels)


# Tracer la courbe ROC pour chaque classe
plt.figure(dpi=300)

for i in range(n_classes):
    # Calculer la courbe ROC pour la classe i (One-vs-Rest)
    fpr_i, tpr_i, _ = roc_curve(y_test == labels[i], y_hat_proba[:, i])  # ROC pour chaque classe
    roc_auc_i = auc(fpr_i, tpr_i)  # Calculer l'AUC pour cette classe

    # Tracer la courbe ROC pour cette classe
    plt.plot(fpr_i, tpr_i, label=f'Classe {labels[i]} (AUC = {roc_auc_i:.2f})')


# Ligne diagonale (référence aléatoire)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Ajout des labels et titre
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbes ROC pour la classification multiclasses')
plt.legend(loc='lower right')


# Sauvegarder l'image avec une haute résolution
plt.savefig("roc_curve_multiclass_smooth.png", dpi=300)

joblib.dump(clf, 'model_with_center_previous_instability_criminality_unemployment_density')

