import pandas as pd
from db_query.query import DatabaseQuery

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# 1. Charger les données via la requête SQL
query = """
    SELECT 
        encodage_avec_centre_gagnant AS prediction,
        encodage_avec_centre_gagnant_precedent AS precedent,
        criminality_per_cir.code_circonscription AS code_circonscription,
        taux_pour_mille,
        taux_chomage,
        density,
        instabilite_avec_centre,
        poids_nuance_avec_centre,
        desir_changement_avec_centre
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir
        ON legislative_per_cir.annee = criminality_per_cir.annee
        AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    INNER JOIN unemployment_per_cir
        ON legislative_per_cir.annee = unemployment_per_cir.annee
        AND legislative_per_cir.code_de_la_circonscription = unemployment_per_cir.code_circonscription
    INNER JOIN density_population
        ON legislative_per_cir.code_du_departement = density_population.code_departement
    WHERE encodage_sans_centre_gagnant IS NOT NULL
      AND encodage_sans_centre_gagnant_precedent IS NOT NULL
"""

# Exécution de la requête et création du DataFrame
results = DatabaseQuery.execute(query, fetch_all=True)
df = pd.DataFrame(results)

# 2. Préparation des features et de la cible
y = df['prediction']
X = df[[
    'precedent',
    'taux_pour_mille',
    'taux_chomage',
    'density',
    'instabilite_avec_centre',
    'poids_nuance_avec_centre',
    'desir_changement_avec_centre'
]]

# 3. Séparation train / test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=808
)

# 4. Construction du Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver='saga',        # supporte l1, l2 et elasticnet
        max_iter=5000,
        class_weight='balanced',
        random_state=808
    ))
])

# 5. Définition de la grille d'hyperparamètres
param_grid = {
    'clf__penalty': ['l2', 'l1', 'elasticnet'],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__l1_ratio': [0.2, 0.5, 0.8]   # utilisé seulement si penalty='elasticnet'
}

# 6. Configuration de la validation croisée stratifiée
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=808
)

# 7. GridSearchCV avec métriques 'roc_auc_ovr' et 'accuracy'
scoring = {
    'roc_auc_ovr': 'roc_auc_ovr',
    'accuracy': 'accuracy'
}
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit='roc_auc_ovr',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 8. Exécution de la recherche
grid_search.fit(X_train, y_train)

# 9. Résultats CV
print("Meilleurs paramètres trouvés :", grid_search.best_params_)
print("Meilleur ROC AUC (OVR) en CV : {:.3f}".format(grid_search.best_score_))
# Métrique Accuracy correspondant au meilleur indice
best_idx = grid_search.best_index_
mean_acc = grid_search.cv_results_['mean_test_accuracy'][best_idx]
print("Accuracy moyenne en CV (pour ces paramètres) : {:.3f}".format(mean_acc))

# 10. Évaluation sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print("\nClassification report sur le test :")
print(classification_report(y_test, y_pred))
print("Test ROC AUC (OVR) : {:.3f}".format(roc_auc_score(y_test, y_proba, multi_class='ovr')))
print("Test Accuracy : {:.3f}".format(accuracy_score(y_test, y_pred)))

# 11. Sauvegarde du pipeline entraîné
joblib.dump(best_model, 'logreg_pipeline_gridsearch.joblib')
print("Pipeline enregistré sous 'logreg_pipeline_gridsearch.joblib'")