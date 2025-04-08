from db_query.query import DatabaseQuery
from google.cloud import bigquery


client = bigquery.Client.from_service_account_json("../mspr-454808-baf9c7d409e4.json")

query_big_query = """
    SELECT 
    code_du_departement, libelle_du_departement, code_de_la_circonscription, inscrits, 
    votants, exprimes, blancs_et_nuls, nuance, voix, nuance_2, voix_2, nuance_3, voix_3, annee, 
    gagnant, voix_gagnant, gagnant_precedent, voix_gagnant_precedent, encodage_sans_centre_nuance, 
    encodage_avec_centre_nuance, encodage_centre_extremes_nuance, encodage_sans_centre_nuance_2, 
    encodage_avec_centre_nuance_2, encodage_centre_extremes_nuance_2, encodage_sans_centre_nuance_3, 
    encodage_avec_centre_nuance_3, encodage_centre_extremes_nuance_3, encodage_sans_centre_gagnant, 
    encodage_avec_centre_gagnant, encodage_centre_extremes_gagnant, encodage_sans_centre_gagnant_precedent, 
    encodage_avec_centre_gagnant_precedent, encodage_centre_extremes_gagnant_precedent 
    FROM `mspr-454808.Legislative.LEG_CIRC_T2_MERGE_HDF_DM`
"""

query_job = client.query(query_big_query)

query_mysql = """
        INSERT INTO predict_election.legislative_per_cir (
            code_du_departement, libelle_du_departement, code_de_la_circonscription, inscrits, 
            votants, exprimes, blancs_et_nuls, nuance, voix, nuance_2, voix_2, nuance_3, voix_3, 
            annee, gagnant, voix_gagnant, gagnant_precedent, voix_gagnant_precedent, 
            encodage_sans_centre_nuance, encodage_avec_centre_nuance, encodage_centre_extremes_nuance, 
            encodage_sans_centre_nuance_2, encodage_avec_centre_nuance_2, encodage_centre_extremes_nuance_2, 
            encodage_sans_centre_nuance_3, encodage_avec_centre_nuance_3, encodage_centre_extremes_nuance_3, 
            encodage_sans_centre_gagnant, encodage_avec_centre_gagnant, encodage_centre_extremes_gagnant, 
            encodage_sans_centre_gagnant_precedent, encodage_avec_centre_gagnant_precedent, 
            encodage_centre_extremes_gagnant_precedent
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
    """

rows_to_insert = []
for row in query_job:
    rows_to_insert.append((
        int(row["code_du_departement"]),
        str(row["libelle_du_departement"]),
        str(row["code_de_la_circonscription"]),
        int(row["inscrits"]),
        int(row["votants"]),
        int(row["exprimes"]),
        int(row["blancs_et_nuls"]),
        str(row["nuance"]),
        int(row["voix"]),
        str(row["nuance_2"]) if row["nuance_2"] is not None else None,
        int(row["voix_2"]) if row["voix_2"] is not None else None,
        str(row["nuance_3"]) if row["nuance_3"] is not None else None,
        int(row["voix_3"]) if row["voix_3"] is not None else None,
        int(row["annee"]),
        str(row["gagnant"]),
        int(row["voix_gagnant"]),
        str(row["gagnant_precedent"]),
        int(row["voix_gagnant_precedent"]),
        int(row["encodage_sans_centre_nuance"]) if row["encodage_sans_centre_nuance"] is not None else None,
        int(row["encodage_avec_centre_nuance"]) if row["encodage_avec_centre_nuance"] is not None else None,
        int(row["encodage_centre_extremes_nuance"]) if row["encodage_centre_extremes_nuance"] is not None else None,
        int(row["encodage_sans_centre_nuance_2"]) if row["encodage_sans_centre_nuance_2"] is not None else None,
        int(row["encodage_avec_centre_nuance_2"]) if row["encodage_avec_centre_nuance_2"] is not None else None,
        int(row["encodage_centre_extremes_nuance_2"]) if row["encodage_centre_extremes_nuance_2"] is not None else None,
        int(row["encodage_sans_centre_nuance_3"]) if row["encodage_sans_centre_nuance_3"] is not None else None,
        int(row["encodage_avec_centre_nuance_3"]) if row["encodage_avec_centre_nuance_3"] is not None else None,
        int(row["encodage_centre_extremes_nuance_3"]) if row["encodage_centre_extremes_nuance_3"] is not None else None,
        int(row["encodage_sans_centre_gagnant"]) if row["encodage_sans_centre_gagnant"] is not None else None,
        int(row["encodage_avec_centre_gagnant"]) if row["encodage_avec_centre_gagnant"] is not None else None,
        int(row["encodage_centre_extremes_gagnant"]) if row["encodage_centre_extremes_gagnant"] is not None else None,
        int(row["encodage_sans_centre_gagnant_precedent"]) if row["encodage_sans_centre_gagnant_precedent"] is not None else None,
        int(row["encodage_avec_centre_gagnant_precedent"]) if row["encodage_avec_centre_gagnant_precedent"] is not None else None,
        int(row["encodage_centre_extremes_gagnant_precedent"]) if row["encodage_centre_extremes_gagnant_precedent"] is not None else None
    ))

DatabaseQuery.execute_many(query_mysql, rows_to_insert)
