import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve, auc, log_loss, f1_score
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
        'instabilite_avec_centre', 'poids_nuance_avec_centre', 'desir_changement_avec_centre', 'is_2024',
        'ratio_menages_proprietaires',
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

learning_rates = [1, 0.6,  0.3, 0.1]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1, 1, 1)

for lr in learning_rates:

    clf = GradientBoostingClassifier(
                    n_estimators=500,
                    max_depth=2,
                    random_state=8,
                    learning_rate=lr
    )
    clf.fit(X_train, y_train)

    scores = np.zeros((clf.n_estimators,), dtype=np.float64)
    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
        scores[i] =  log_loss(y_test, y_proba, labels=labels)

    ax.plot(
        (np.arange(scores.shape[0]) + 1),
        scores,
        "-",
        label=f"alpha: {lr}",
    )

ax.grid(True, which = 'both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("Itérations")
ax.set_ylabel("Log Loss (test)")
ax.set_title("Influence du learning rate sur la performance du GradientBoosting")
ax.legend()
plt.tight_layout()
plt.savefig('influence_learning_rate_performance_gradient_boosting.png')

clf = GradientBoostingClassifier(
                    n_estimators=500,
                    max_depth=3,
                    random_state=8,
                    learning_rate=0.1
    )
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

precision, recall, thresholds = precision_recall_curve(y_test == 1, y_hat_proba[:, 0])  # classe 1

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Seuil de probabilité")
plt.legend()
plt.title("Courbe Precision-Recall pour classe 1")
plt.grid(True)
plt.savefig('Courbe Precision-Recall pour classe 1')

# On récupère les seuils, précisions, rappels
precision, recall, thresholds = precision_recall_curve(y_test == 1, y_hat_proba[:, 0])

# Calcul du F1-score à chaque seuil
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)


# Binarisation des labels pour classification One-vs-Rest
y_test_bin = label_binarize(y_test, classes=labels)

# Stocker les meilleurs seuils pour chaque classe
best_thresholds = {}
# Trouver le meilleur seuil par classe (celui qui maximise le F1-score)
for i, label in enumerate(labels):
    precision, recall, thresholds = precision_recall_curve(y_test_bin[:, i], y_hat_proba[:, i])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_thresholds[label] = best_threshold
    print(f"✅ Classe {label} — Meilleur seuil : {best_threshold:.3f} | F1 : {f1_scores[best_index]:.3f}")


# Fonction pour appliquer les seuils personnalisés
def predict_with_thresholds(probas, thresholds_dict):
    custom_preds = []
    for proba in probas:
        # Appliquer le seuil sur chaque probabilité
        scores = []
        for i, label in enumerate(labels):
            score = proba[i] - thresholds_dict[label]  # Score net au-dessus du seuil
            scores.append(score)
        # Sélection de la classe avec le score le plus élevé
        custom_preds.append(labels[np.argmax(scores)])
    return np.array(custom_preds)

# Prédictions finales avec seuils optimisés
y_pred_custom = predict_with_thresholds(y_hat_proba, best_thresholds)

# Matrice de confusion et scores
print("✅ Résultats avec seuils personnalisés :")
print("Accuracy :", accuracy_score(y_test, y_pred_custom))
print("Precision :", precision_score(y_test, y_pred_custom, average="weighted", zero_division=1))
print("Recall :", recall_score(y_test, y_pred_custom, average="weighted", zero_division=1))
print("F1 Score :", f1_score(y_test, y_pred_custom, average="weighted"))

conf_matrix_thresh = confusion_matrix(y_test, y_pred_custom)
ConfusionMatrixDisplay(conf_matrix_thresh, display_labels=labels).plot()
plt.title("Matrice de confusion avec seuils personnalisés")
plt.savefig("confusion_matrix_thresholds.png")
