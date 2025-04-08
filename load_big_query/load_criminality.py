from db_query.query import DatabaseQuery
from google.cloud import bigquery


client = bigquery.Client.from_service_account_json("../mspr-454808-d750cb51f2b1.json")

query_big_query = """
    SELECT code_de_la_circonscription, annee, nombre, population, taux_pour_mille 
    FROM `mspr-454808.Criminality.CRIMINALITY_PER_CIRC_HDF_DM`
"""
query_job = client.query(query_big_query)

query_mysql = """
        INSERT INTO predict_election.criminality_per_cir
        (code_circonscription, annee, nombre, population, taux_pour_mille)
        VALUES (%s, %s, %s, %s, %s)
    """

rows_to_insert = []
for row in query_job:
    rows_to_insert.append((
        str(row["code_de_la_circonscription"]),
        int(row["annee"]),
        int(row["nombre"]),
        int(row["population"]),
        float(row["taux_pour_mille"])
    ))


DatabaseQuery.execute_many(query_mysql, rows_to_insert)
