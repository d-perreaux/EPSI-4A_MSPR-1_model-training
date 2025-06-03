import plotly.graph_objects as go

from db_query.query import DatabaseQuery

labels_amount = 5
labels = [
    # 2024
    "Extrême gauche", "Gauche", "Centre", "Droite", "Extrême droite",
    # 2022
    "Extrême gauche", "Gauche", "Centre", "Droite", "Extrême droite",
    # 2017
    "Extrême gauche", "Gauche", "Centre", "Droite", "Extrême droite",
    # 2012
    "Extrême gauche", "Gauche", "Centre", "Droite", "Extrême droite"
]


# Dictionnaire pour mapper les sources à leur index
label_index = {label: i for i, label in enumerate(labels)}

print(label_index)

results_select_2024 = """
    SELECT 
    encodage_centre_extremes_gagnant, encodage_centre_extremes_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_centre_extremes_gagnant IS NOT NULL
    AND encodage_centre_extremes_gagnant_precedent IS NOT NULL
    AND annee = 2024
    """
source_2024 = {}
results_job_2024 = DatabaseQuery.execute(results_select_2024, fetch_all=True)
print(results_job_2024)
for row in results_job_2024:
    if row['encodage_centre_extremes_gagnant'] not in source_2024:
        source_2024[row['encodage_centre_extremes_gagnant']] = {}
    if row['encodage_centre_extremes_gagnant_precedent'] not in source_2024[row['encodage_centre_extremes_gagnant']]:
        source_2024[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] = 0
    source_2024[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] += 1

print(source_2024)

results_select_2022 = """
    SELECT 
    encodage_centre_extremes_gagnant, encodage_centre_extremes_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_centre_extremes_gagnant IS NOT NULL
    AND encodage_centre_extremes_gagnant_precedent IS NOT NULL
    AND annee = 2022
    """
source_2022 = {}
results_job_2022 = DatabaseQuery.execute(results_select_2022, fetch_all=True)
print(results_job_2022)
for row in results_job_2022:
    if row['encodage_centre_extremes_gagnant'] not in source_2022:
        source_2022[row['encodage_centre_extremes_gagnant']] = {}
    if row['encodage_centre_extremes_gagnant_precedent'] not in source_2022[row['encodage_centre_extremes_gagnant']]:
        source_2022[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] = 0
    source_2022[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] += 1

print(source_2022)

results_select_2017 = """
    SELECT 
    encodage_centre_extremes_gagnant, encodage_centre_extremes_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_centre_extremes_gagnant IS NOT NULL
    AND encodage_centre_extremes_gagnant_precedent IS NOT NULL
    AND annee = 2017
    """
source_2017 = {}
results_job_2017 = DatabaseQuery.execute(results_select_2017, fetch_all=True)
print(results_job_2017)
for row in results_job_2017:
    if row['encodage_centre_extremes_gagnant'] not in source_2017:
        source_2017[row['encodage_centre_extremes_gagnant']] = {}
    if row['encodage_centre_extremes_gagnant_precedent'] not in source_2017[row['encodage_centre_extremes_gagnant']]:
        source_2017[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] = 0
    source_2017[row['encodage_centre_extremes_gagnant']][row['encodage_centre_extremes_gagnant_precedent']] += 1

print(source_2017)


# Étape 3 : construire les liens source → target → value
source = []
target = []
value = []

# Lecture : 2024 à gauche et on remonte le temps
for winner, dict_previous in source_2024.items():
    for previous, count in dict_previous.items():
        source.append(winner - 1)
        target.append(previous + 4)
        value.append(count)

for winner, dict_previous in source_2022.items():
    for previous, count in dict_previous.items():
        source.append(winner + 4)
        target.append(previous + 9)
        value.append(count)

for winner, dict_previous in source_2017.items():
    for previous, count in dict_previous.items():
        source.append(winner + 9)
        target.append(previous + 14)
        value.append(count)

# Lecture : 2012 à gauche et on va vers 2024
# # 2012 → 2017
# for winner, dict_previous in source_2017.items():
#     for previous, count in dict_previous.items():
#         source.append(previous + 0)      # 2012 index: 1–7
#         target.append(winner + 5)        # 2017 index: 5–8
#         value.append(count)
#
# # 2017 → 2022
# for winner, dict_previous in source_2022.items():
#     for previous, count in dict_previous.items():
#         source.append(previous + 5)      # 2017 index: 5–8
#         target.append(winner + 9)        # 2022 index: 9–12
#         value.append(count)
#
# # 2022 → 2024
# for winner, dict_previous in source_2024.items():
#     for previous, count in dict_previous.items():
#         source.append(previous + 9)      # 2022 index: 9–12
#         target.append(winner + 13)       # 2024 index: 13–16
#         value.append(count)

print(source)
print(target)
print(value)

color_mapping = {
    "Extrême gauche": "#ea5962",  # par exemple, rouge clair
    "Gauche": "#e7bfb9",         # bleu clair
    "Centre": "#ffff00",         # jaune
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

fig.update_layout(title_text="Quelle était la nuance précédente de chaque circonscription ? (2024 → 2022 → 2017 → 2012)", font_size=12)
# Années et positions horizontales relatives (xref='paper')
years = ["2024", "2022", "2017", "2012"]
x_positions = [-0.01, 0.32, 0.66, 1.00]   # à ajuster en fonction de vos marges

# Crée les annotations
annotations = []
for x, year in zip(x_positions, years):
    annotations.append(dict(
        x=x, y=1.05,                    # un petit peu au-dessus de la zone de dessin
        xref='paper', yref='paper',
        text=year,
        showarrow=False,
        font=dict(size=14, color="#444")
    ))

# Applique layout
fig.update_layout(
    margin=dict(t=80, l=50, r=50, b=50),
    annotations=annotations
)

fig.write_html("sankey_diagram_previous_shade_centre_extremes.html")
