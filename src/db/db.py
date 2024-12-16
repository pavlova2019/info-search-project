import sqlite3
import src.config as cfg


def setup_database(path: str = cfg.RATINGS_DB_PATH):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            message_id INTEGER,
            query TEXT,
            response TEXT,
            rating TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_rating(user_id: int, message_id: int, query: str, response: str,
                rating: str, path: str = cfg.RATINGS_DB_PATH):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ratings (user_id, message_id, query, response, rating)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, message_id, query, response, rating))
    conn.commit()
    conn.close()
    