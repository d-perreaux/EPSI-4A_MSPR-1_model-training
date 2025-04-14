import plotly.graph_objects as go

from db_query.query import DatabaseQuery

labels_amount = 4
labels = [
    # 2024
    "Extrême gauche", "Gauche", "Droite", "Extrême droite",
    # 2022
    "Extrême gauche", "Gauche", "Droite", "Extrême droite",
    # 2017
    "Extrême gauche", "Gauche", "Droite", "Extrême droite",
    # 2012
    "Extrême gauche", "Gauche", "Droite", "Extrême droite"
]


# Dictionnaire pour mapper les sources à leur index
label_index = {label: i for i, label in enumerate(labels)}

print(label_index)

results_select_2024 = """
    SELECT 
    encodage_sans_centre_gagnant, encodage_sans_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
    AND annee = 2024
    """
source_2024 = {}
results_job_2024 = DatabaseQuery.execute(results_select_2024, fetch_all=True)
print(results_job_2024)
for row in results_job_2024:
    if row['encodage_sans_centre_gagnant'] not in source_2024:
        source_2024[row['encodage_sans_centre_gagnant']] = {}
    if row['encodage_sans_centre_gagnant_precedent'] not in source_2024[row['encodage_sans_centre_gagnant']]:
        source_2024[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] = 0
    source_2024[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] += 1

print(source_2024)

results_select_2022 = """
    SELECT 
    encodage_sans_centre_gagnant, encodage_sans_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
    AND annee = 2022
    """
source_2022 = {}
results_job_2022 = DatabaseQuery.execute(results_select_2022, fetch_all=True)
print(results_job_2022)
for row in results_job_2022:
    if row['encodage_sans_centre_gagnant'] not in source_2022:
        source_2022[row['encodage_sans_centre_gagnant']] = {}
    if row['encodage_sans_centre_gagnant_precedent'] not in source_2022[row['encodage_sans_centre_gagnant']]:
        source_2022[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] = 0
    source_2022[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] += 1

print(source_2022)

results_select_2017 = """
    SELECT 
    encodage_sans_centre_gagnant, encodage_sans_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_sans_centre_gagnant IS NOT NULL
    AND encodage_sans_centre_gagnant_precedent IS NOT NULL
    AND annee = 2017
    """
source_2017 = {}
results_job_2017 = DatabaseQuery.execute(results_select_2017, fetch_all=True)
print(results_job_2017)
for row in results_job_2017:
    if row['encodage_sans_centre_gagnant'] not in source_2017:
        source_2017[row['encodage_sans_centre_gagnant']] = {}
    if row['encodage_sans_centre_gagnant_precedent'] not in source_2017[row['encodage_sans_centre_gagnant']]:
        source_2017[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] = 0
    source_2017[row['encodage_sans_centre_gagnant']][row['encodage_sans_centre_gagnant_precedent']] += 1

print(source_2017)


# Étape 3 : construire les liens source → target → value
source = []
target = []
value = []

for winner, dict_previous in source_2024.items():
    for previous, count in dict_previous.items():
        source.append(winner - 1)
        target.append(previous + 3)
        value.append(count)

for winner, dict_previous in source_2022.items():
    for previous, count in dict_previous.items():
        source.append(winner + 3)
        target.append(previous + 7)
        value.append(count)

for winner, dict_previous in source_2017.items():
    for previous, count in dict_previous.items():
        source.append(winner + 7)
        target.append(previous + 11)
        value.append(count)

print(source)
print(target)
print(value)

color_mapping = {
    "Extrême gauche": "#ea5962",  # par exemple, rouge clair
    "Gauche": "#e7bfb9",         # bleu clair
    "Droite": "#52b4e3",         # vert clair
    "Extrême droite": "#031a47"  # orange clair
}

link_colors = []
for s in source:
    node_label = labels[s]
    link_colors.append(color_mapping[node_label])

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="lightgray"
    ),
    # évite que les colonnes se mélangent,
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
)])

fig.update_layout(title_text="Évolution des nuances politiques (2024 → 2022 → 2017 -> 2012)", font_size=12)
fig.write_html("sankey_diagram_previous_shade.html")
