import math
from db_query.query import DatabaseQuery


code_circumscription_select = """
    SELECT DISTINCT
    code_de_la_circonscription
    FROM prediction_2027
    ORDER BY code_de_la_circonscription ASC
"""
code_circumscription_job = DatabaseQuery.execute(code_circumscription_select, fetch_all=True)
codes_circumscription = []

for row in code_circumscription_job:
    codes_circumscription.append(row['code_de_la_circonscription'])

years_select = """
    SELECT DISTINCT
    annee
    FROM legislative_per_cir
    ORDER BY annee DESC
"""
years_job = DatabaseQuery.execute(years_select, fetch_all=True)
years = []

for row in years_job:
    years.append(row['annee'])

years_to_process = [2024]


results_select = """
    SELECT code_de_la_circonscription, annee,
    encodage_sans_centre_gagnant, encodage_sans_centre_gagnant_precedent,
    encodage_avec_centre_gagnant, encodage_avec_centre_gagnant_precedent,
    encodage_centre_extremes_gagnant, encodage_centre_extremes_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
    AND encodage_avec_centre_gagnant IS NOT NULL
    AND encodage_avec_centre_gagnant_precedent IS NOT NULL
    AND encodage_centre_extremes_gagnant IS NOT NULL
    AND encodage_centre_extremes_gagnant_precedent IS NOT NULL
"""
results_job = DatabaseQuery.execute(results_select, fetch_all=True)

# Start the instability algorithm
for code_circumscription in codes_circumscription:
    compteur_annee = 0
    for i, year in enumerate(years_to_process):
        for result in results_job:
            if (result['annee'] == year) and (result['code_de_la_circonscription'] == code_circumscription):

                instability = 0
                weight_nuance = 0
                rise_change = 0

                actual_winner = result['encodage_avec_centre_gagnant']
                winner_n_1 = next((r['encodage_avec_centre_gagnant'] for r in results_job if r['annee'] == (years[i+1]) and r['code_de_la_circonscription'] == code_circumscription))
                winner_n_2 = next((r['encodage_avec_centre_gagnant'] for r in results_job if
                                   r['annee'] == (years[i + 2]) and r[
                                       'code_de_la_circonscription'] == code_circumscription))
                winner_n_3 = next((r['encodage_avec_centre_gagnant'] for r in results_job if
                                   r['annee'] == (years[i + 3]) and r[
                                       'code_de_la_circonscription'] == code_circumscription))
                winner_n_4 = next((r['encodage_avec_centre_gagnant'] for r in results_job if
                                   r['annee'] == (years[i + 4]) and r[
                                       'code_de_la_circonscription'] == code_circumscription))

                # Instability
                if actual_winner != winner_n_1:
                    instability += 3
                if winner_n_1 != winner_n_2:
                    instability += 2
                if winner_n_2 != winner_n_3:
                    instability += 1

                # Weight nuance n-1 in the circumscription
                if actual_winner == winner_n_1:
                    weight_nuance += 3
                if actual_winner == winner_n_2:
                    weight_nuance += 2
                if actual_winner == winner_n_3:
                    weight_nuance += 1

                # Rise of change desir
                if actual_winner == winner_n_1:
                    rise_change += 1
                    if actual_winner == winner_n_2:
                        rise_change += 1
                        if actual_winner == winner_n_3:
                            rise_change += 1

                print(f'{code_circumscription} - {year}')
                print(f'Instability - {instability}')
                print(f'Weight nuance - {weight_nuance}')
                print(f'Change desir - {rise_change}')
                print('\n')

                query_insert = """
                UPDATE prediction_2027
                SET instabilite_avec_centre = %s, poids_nuance_avec_centre = %s, desir_changement_avec_centre = %s
                WHERE code_de_la_circonscription = %s
                """
                params = (instability, weight_nuance, rise_change, code_circumscription)

                DatabaseQuery.execute(query_insert, params)

                compteur_annee += 1
