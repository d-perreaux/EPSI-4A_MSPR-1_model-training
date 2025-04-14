import plotly.graph_objects as go

# Étape 1 : liste ordonnée des labels (nuances politiques par année)
labels = [
    # 2017
    "LREM_2017", "LR_2017", "PS_2017",
    # 2022
    "Renaissance_2022", "LR_2022", "NUPES_2022",
    # 2024
    "Renaissance_2024", "LR_2024", "NouveauParti_2024"
]

# Étape 2 : dictionnaire pour accéder facilement à l'indice d'un label
label_index = {label: i for i, label in enumerate(labels)}

# Étape 3 : construire les liens source → target → value
source = []
target = []
value = []

# 2017 vers 2022
source.append(label_index["LREM_2017"])
target.append(label_index["Renaissance_2022"])
value.append(40)

source.append(label_index["LR_2017"])
target.append(label_index["LR_2022"])
value.append(30)

source.append(label_index["PS_2017"])
target.append(label_index["NUPES_2022"])
value.append(25)

# 2022 vers 2024
source.append(label_index["Renaissance_2022"])
target.append(label_index["Renaissance_2024"])
value.append(35)

source.append(label_index["LR_2022"])
target.append(label_index["LR_2024"])
value.append(25)

source.append(label_index["NUPES_2022"])
target.append(label_index["NouveauParti_2024"])
value.append(20)

# Étape 4 : construction du Sankey avec Plotly
fig = go.Figure(data=[go.Sankey(
    arrangement = "snap",  # évite que les colonnes se mélangent
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = labels,
        color = "skyblue"
    ),
    link = dict(
        source = source,
        target = target,
        value = value,
        color = ["#a0c4ff", "#ffc6ff", "#bdb2ff", "#a0c4ff", "#ffc6ff", "#bdb2ff"]
    )
)])

fig.update_layout(title_text="Évolution des nuances politiques (2017 → 2022 → 2024)", font_size=12)
fig.write_html("sankey_test_chatou.html")
