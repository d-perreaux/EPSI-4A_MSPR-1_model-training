from db_connect.connection import DatabaseConnection


class DatabaseQuery:
    @staticmethod
    def execute(sql, params=None, fetch_one=False, fetch_all=False):
        with DatabaseConnection() as conn:
            if not conn:
                print(f"DatabaseQuery.execute(): Pas de connexion à la base de données")
                return None

            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(sql, params or ())

                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else:
                    conn.commit()
                    result = cursor.rowcount

                cursor.close()
                return result
            except Exception as e:
                print(f"Erreur SQL : {e}")
                return None

    @staticmethod
    def execute_many(sql, params_list):
        with DatabaseConnection() as conn:
            if not conn:
                print("Pas de connexion à la base de données.")
                return None

            try:
                cursor = conn.cursor()
                cursor.executemany(sql, params_list)
                conn.commit()
                rowcount = cursor.rowcount
                cursor.close()
                return rowcount
            except Exception as e:
                print(f"Erreur SQL : {e}")
                return None
