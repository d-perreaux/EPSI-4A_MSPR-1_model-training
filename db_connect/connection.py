import mysql.connector
import os

from dotenv import load_dotenv


load_dotenv()


class DatabaseConnection:

    def __init__(self):
        self.conn = None

    def __enter__(self):
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                use_pure=True
            )
            return self.conn
        except Exception as e:
            print(f"Erreur de connexion: {e}")
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn and self.conn.is_connected():
            self.conn.close()
