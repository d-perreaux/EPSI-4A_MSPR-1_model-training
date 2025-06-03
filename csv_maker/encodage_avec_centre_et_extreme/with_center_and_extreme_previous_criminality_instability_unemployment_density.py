import csv

from db_query.query import DatabaseQuery

query = """
    SELECT 
    encodage_centre_extremes_gagnant as prediction,
    encodage_centre_extremes_gagnant_precedent as precedent,
    criminality_per_cir.code_circonscription as code_circonscirption,
    taux_pour_mille, taux_chomage, density, instabilite_centre_extremes,
    poids_nuance_centre_extremes, desir_changement_centre_extremes 
    FROM legislative_per_cir
    INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee 
        AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
    INNER JOIN unemployment_per_cir ON legislative_per_cir.annee = unemployment_per_cir.annee 
        AND legislative_per_cir.code_de_la_circonscription = unemployment_per_cir.code_circonscription
    INNER JOIN density_population ON legislative_per_cir.code_du_departement = density_population.code_departement
    WHERE encodage_centre_extremes_gagnant IS NOT NULL
    AND encodage_centre_extremes_gagnant_precedent IS NOT NULL
"""

results = DatabaseQuery.execute(query, fetch_all=True)

if results:
    # noms de colonnes depuis le premier resultat du dico
    fieldnames = results[0].keys()
    with open("with_center_and_extreme_previous_instability_criminality_unemployment_density.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Fichier CSV généré avec succès.")
else:
    print("Aucune donnée à écrire.")
