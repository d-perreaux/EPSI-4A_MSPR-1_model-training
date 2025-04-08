import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from db_query.query import DatabaseQuery

query = """
    SELECT 
    code_circonscription,
    taux_pour_mille, annee, departement, density 
    FROM criminality_per_cir
    INNER JOIN density_population ON density_population.code_departement = criminality_per_cir.departement
"""
query_job = DatabaseQuery.execute(query, fetch_all=True)

df = pd.DataFrame(query_job)
df['criminality_density'] = (df['taux_pour_mille'] / df['density'])

plt.figure(figsize=(16, 8))
sns.boxplot(x='code_circonscription', y='criminality_density', data=df)
plt.xticks(rotation=90)
plt.title('Dispersion de la criminalité par circonscription (2016-2022)')
plt.xlabel("Circonscription")
plt.ylabel("criminality_density")
plt.tight_layout()
plt.savefig("boxplot_circonscription_global_criminality_density.png")

for dept in df['departement'].dropna().unique():
    df_dept = df[df['departement'] == dept]

    mean_dept = df_dept['criminality_density'].mean()
    q1 = np.percentile(df_dept['criminality_density'], 25)
    median = np.percentile(df_dept['criminality_density'], 50)
    q3 = np.percentile(df_dept['criminality_density'], 75)
    min_val = df_dept['criminality_density'].min()
    max_val = df_dept['criminality_density'].max()
    iqr = q3 - q1

    plt.figure(figsize=(16, 8))
    sns.boxplot(x='code_circonscription', y='criminality_density', data=df_dept)
    sns.boxplot(x=[dept] * len(df_dept), y='criminality_density', data=df_dept, color='orange', width=0.5)

    plt.text(x=0.95, y=max_val + 0.5,
             s=f"Min: {min_val:.2f}, Q1: {q1:.2f}, Median: {median:.2f}, Q3: {q3:.2f}, Max: {max_val:.2f}",
             color='black', fontsize=10, ha='center', va='bottom')

    plt.xticks(rotation=90)
    plt.title(f'Dispersion de la criminality_density - Département {dept} (2016-2022) - Moyenne: {mean_dept:.2f}')
    plt.xlabel("Circonscription")
    plt.ylabel("criminality_density")
    plt.tight_layout()

    plt.savefig(f"boxplots_par_departement_{dept}.png")

    query_select = """
        SELECT 
        encodage_avec_centre_gagnant as prediction, encodage_avec_centre_gagnant_precedent as precedent, criminality_per_cir.code_circonscription as code_circonscription,
        taux_pour_mille, criminality_per_cir.annee as annee, density
        FROM legislative_per_cir
        INNER JOIN criminality_per_cir ON legislative_per_cir.annee = criminality_per_cir.annee AND legislative_per_cir.code_de_la_circonscription = criminality_per_cir.code_circonscription
        INNER JOIN density_population ON density_population.code_departement = criminality_per_cir.departement
        WHERE encodage_avec_centre_gagnant_precedent IS NOT NULL
        AND encodage_avec_centre_gagnant IS NOT NULL
        AND departement = %s
    """
    query_job = DatabaseQuery.execute(query_select, (dept, ), fetch_all=True)
    df_predict = pd.DataFrame(query_job)
    df_predict['criminality_density'] = (df_predict['taux_pour_mille'] / df_predict['density'])

    plt.figure()
    sns.scatterplot(x='code_circonscription', y='criminality_density', data=df_predict, hue='prediction')
    plt.savefig(f'criminality_density_scatterplot_{dept}.png')
