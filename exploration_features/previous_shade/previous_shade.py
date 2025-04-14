from db_query.query import DatabaseQuery


code_circumscription_select = """
    SELECT DISTINCT
    code_de_la_circonscription
    FROM legislative_per_cir
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

# Check if all lines are ok
for code_circumscription in codes_circumscription:
    compteur_annee = 0
    for i, year in enumerate(years):
        if i == len(years) - 1:
            break
        else:
            for result in results_job:
                if (result['annee'] == year) and (result['code_de_la_circonscription'] == code_circumscription):
                    actual_winner = result['encodage_sans_centre_gagnant']
                    winner_n_1 = next((r['encodage_sans_centre_gagnant'] for r in results_job if r['annee'] == (years[i+1]) and r['code_de_la_circonscription'] == code_circumscription))
                    # print(f'{code_circumscription} - {year} : gagnant actuel {actual_winner} - gagnant précédent {winner_n_1} ')
                    compteur_annee += 1
    if compteur_annee != 6:
        print(code_circumscription)

shade_dictionnary = {}

for result in results_job:
    if result['encodage_sans_centre_gagnant'] not in shade_dictionnary:
        shade_dictionnary[result['encodage_sans_centre_gagnant']] = {}

    if result['encodage_sans_centre_gagnant_precedent'] not in shade_dictionnary[result['encodage_sans_centre_gagnant']]:
        shade_dictionnary[result['encodage_sans_centre_gagnant']][result['encodage_sans_centre_gagnant_precedent']] = 0

    shade_dictionnary[result['encodage_sans_centre_gagnant']][result['encodage_sans_centre_gagnant_precedent']] += 1

for winner in shade_dictionnary:
    print(winner)
    for previous in shade_dictionnary[winner]:
        print(f'{previous} : {shade_dictionnary[winner][previous]}')