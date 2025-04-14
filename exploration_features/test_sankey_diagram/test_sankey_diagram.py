import plotly.graph_objects as go


source = [0, 0, 1, 1, 0]
target = [2, 3, 4, 5, 4]
value = [8, 2, 2, 8, 4]

link = dict(source=source, target=target, value=value)

data = go.Sankey(link=link)
fig = go.Figure(data)

fig.write_html("sankey_test.html")
