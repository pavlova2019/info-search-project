import os
import sqlite3

def setup_database(path: str):
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


def save_rating(user_id: int, message_id: int, query: str,
                response: str, rating: str, path: str):
    if not os.path.exists(path):
        setup_database(path)
        
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ratings (user_id, message_id, query, response, rating)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, message_id, query, response, rating))
    conn.commit()
    conn.close()
