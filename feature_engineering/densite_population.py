from db_query.query import DatabaseQuery


query_select = """
    SELECT 
    code_circonscription,
    id 
    FROM criminality_per_cir
"""
query_job = DatabaseQuery.execute(query_select, fetch_all=True)

query_add = """
    UPDATE criminality_per_cir
    SET departement = %s
    WHERE id = %s
"""

for row in query_job:
    departement = row['code_circonscription'][:2]
    departement_tuple = (departement, row['id'])
    query_job = DatabaseQuery.execute(query_add, departement_tuple)
