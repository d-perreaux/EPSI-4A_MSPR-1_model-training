from db_query.query import DatabaseQuery
from google.cloud import bigquery


client = bigquery.Client.from_service_account_json("../mspr-454808-d750cb51f2b1.json")

query_big_query = """
    SELECT code_de_la_circonscription, annee, population, chomeurs, taux_chomage 
    FROM `mspr-454808.Unemployment.UNEMPLOYMENT_PER_CIRC_HDF_DW`
"""
query_job = client.query(query_big_query)

query_mysql = """
        INSERT INTO predict_election.unemployment_per_cir
        (code_circonscription, annee, population, chomeurs, taux_chomage)
        VALUES (%s, %s, %s, %s, %s)
    """

rows_to_insert = []
for row in query_job:
    rows_to_insert.append((
        str(row["code_du_departement"]),
        int(row["annee"]),
        int(row["population"]),
        int(row["chomeurs"]),
        float(row["taux_chomage"])
    ))


DatabaseQuery.execute_many(query_mysql, rows_to_insert)
