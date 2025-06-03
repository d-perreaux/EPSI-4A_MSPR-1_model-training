import plotly.graph_objects as go

from db_query.query import DatabaseQuery

labels_amount = 3
labels = [
    # 2024
    "Gauche", "Centre", "Droite",
    # 2022
    "Gauche", "Centre", "Droite",
    # 2017
    "Gauche", "Centre", "Droite",
    # 2012
    "Gauche", "Centre", "Droite"
]


# Dictionnaire pour mapper les sources à leur index
label_index = {label: i for i, label in enumerate(labels)}

print(label_index)

results_select_2024 = """
    SELECT 
    encodage_avec_centre_gagnant, encodage_avec_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_avec_centre_gagnant IS NOT NULL
    AND encodage_avec_centre_gagnant_precedent IS NOT NULL
    AND annee = 2024
    """
source_2024 = {}
results_job_2024 = DatabaseQuery.execute(results_select_2024, fetch_all=True)
print(results_job_2024)
for row in results_job_2024:
    if row['encodage_avec_centre_gagnant'] not in source_2024:
        source_2024[row['encodage_avec_centre_gagnant']] = {}
    if row['encodage_avec_centre_gagnant_precedent'] not in source_2024[row['encodage_avec_centre_gagnant']]:
        source_2024[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] = 0
    source_2024[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] += 1

print(source_2024)

results_select_2022 = """
    SELECT 
    encodage_avec_centre_gagnant, encodage_avec_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_avec_centre_gagnant IS NOT NULL
    AND encodage_avec_centre_gagnant_precedent IS NOT NULL
    AND annee = 2022
    """
source_2022 = {}
results_job_2022 = DatabaseQuery.execute(results_select_2022, fetch_all=True)
print(results_job_2022)
for row in results_job_2022:
    if row['encodage_avec_centre_gagnant'] not in source_2022:
        source_2022[row['encodage_avec_centre_gagnant']] = {}
    if row['encodage_avec_centre_gagnant_precedent'] not in source_2022[row['encodage_avec_centre_gagnant']]:
        source_2022[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] = 0
    source_2022[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] += 1

print(source_2022)

results_select_2017 = """
    SELECT 
    encodage_avec_centre_gagnant, encodage_avec_centre_gagnant_precedent
    FROM legislative_per_cir
    WHERE encodage_avec_centre_gagnant IS NOT NULL
    AND encodage_avec_centre_gagnant_precedent IS NOT NULL
    AND annee = 2017
    """
source_2017 = {}
results_job_2017 = DatabaseQuery.execute(results_select_2017, fetch_all=True)
print(results_job_2017)
for row in results_job_2017:
    if row['encodage_avec_centre_gagnant'] not in source_2017:
        source_2017[row['encodage_avec_centre_gagnant']] = {}
    if row['encodage_avec_centre_gagnant_precedent'] not in source_2017[row['encodage_avec_centre_gagnant']]:
        source_2017[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] = 0
    source_2017[row['encodage_avec_centre_gagnant']][row['encodage_avec_centre_gagnant_precedent']] += 1

print(source_2017)


# Étape 3 : construire les liens source → target → value
source = []
target = []
value = []

# Lecture : 2024 à gauche et on remonte le temps
for winner, dict_previous in source_2024.items():
    for previous, count in dict_previous.items():
        source.append(winner - 1)
        target.append(previous + 2)
        value.append(count)

for winner, dict_previous in source_2022.items():
    for previous, count in dict_previous.items():
        source.append(winner + 2)
        target.append(previous + 5)
        value.append(count)

for winner, dict_previous in source_2017.items():
    for previous, count in dict_previous.items():
        source.append(winner + 5)
        target.append(previous + 8)
        value.append(count)


print(source)
print(target)
print(value)

color_mapping = {
    "Gauche": "#ea5962",  # rouge
    "Centre": "#ffff00",  # jaune
    "Droite": "#4d93d9"  # bleu
}

link_colors = []
for s in source:
    node_label = labels[s]
    link_colors.append(color_mapping[node_label])

# calcul des positions fixes
n_cols = 4
# colonnes uniformément réparties de x=0 à x=1
xs = [i/(n_cols-1) for i in range(n_cols)]
# positions y pour chaque nuance
y_map = {"Droite": 0.1, "Centre": 0.5, "Gauche": 0.9}

node_x = []
node_y = []
for col in range(n_cols):
    for shade in ["Gauche", "Centre", "Droite"]:
        node_x.append(xs[col])
        node_y.append(y_map[shade])

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="lightgray",
        x=node_x,
        y=node_y
    ),
    # évite que les colonnes se mélangent,
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
)])

fig.update_layout(title_text="Évolution des nuances politiques (2024 → 2022 → 2017 → 2012)", font_size=12)

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

fig.write_html("sankey_diagram_previous_shade_avec_centre.html")
