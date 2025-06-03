import csv

from db_query.query import DatabaseQuery

query = """
    SELECT 
    encodage_avec_centre_gagnant as prediction,
    encodage_avec_centre_gagnant_precedent as precedent,
    criminality_per_cir.code_circonscription as code_circonscirption,
    taux_pour_mille, taux_chomage, density, instabilite_avec_centre,
    poids_nuance_avec_centre, desir_changement_avec_centre 
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee 
        AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    INNER JOIN unemployment_per_cir ON legislative_per_cir.annee = unemployment_per_cir.annee 
        AND legislative_per_cir.code_de_la_circonscription = unemployment_per_cir.code_circonscription
    INNER JOIN density_population ON legislative_per_cir.code_du_departement = density_population.code_departement
    WHERE encodage_avec_centre_gagnant IS NOT NULL
    AND encodage_avec_centre_gagnant_precedent IS NOT NULL
"""

results = DatabaseQuery.execute(query, fetch_all=True)
print(results)

if results:
    # noms de colonnes depuis le premier resultat du dico
    fieldnames = results[0].keys()
    with open("without_center_previous_instability_criminality_unemployment_density.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Fichier CSV généré avec succès.")
else:
    print("Aucune donnée à écrire.")
