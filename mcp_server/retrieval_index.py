import os
import sqlite3
from typing import Dict, List, Tuple, Any

from config import LEXICAL_INDEX_DB_PATH


class RetrievalIndex:
    def __init__(self, db_path: str = LEXICAL_INDEX_DB_PATH):
        self.db_path = db_path
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_index (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    text_content TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS file_index_fts
                USING fts5(path, text_content, tokenize='unicode61');
                """
            )
            conn.commit()
            self._initialized = True
        finally:
            conn.close()

    def upsert_files(self, file_payloads: List[Dict[str, Any]]) -> None:
        if not file_payloads:
            return
        self.ensure_initialized()
        conn = self._connect()
        try:
            for item in file_payloads:
                path = item["path"]
                mtime = float(item["mtime"])
                text_content = item["text_content"]

                row = conn.execute("SELECT mtime FROM file_index WHERE path = ?", (path,)).fetchone()
                if row and float(row["mtime"]) == mtime:
                    continue

                conn.execute(
                    """
                    INSERT INTO file_index(path, mtime, text_content)
                    VALUES (?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        mtime = excluded.mtime,
                        text_content = excluded.text_content
                    """,
                    (path, mtime, text_content),
                )
                conn.execute("DELETE FROM file_index_fts WHERE path = ?", (path,))
                conn.execute(
                    "INSERT INTO file_index_fts(path, text_content) VALUES (?, ?)",
                    (path, text_content),
                )
            conn.commit()
        finally:
            conn.close()

    def query(self, query_text: str, limit: int = 25) -> List[Tuple[str, float]]:
        self.ensure_initialized()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT path, bm25(file_index_fts) AS score
                FROM file_index_fts
                WHERE file_index_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query_text, limit),
            ).fetchall()
            return [(str(r["path"]), float(r["score"])) for r in rows]
        except sqlite3.OperationalError:
            # malformed query or tokenizer edge case
            return []
        finally:
            conn.close()

    def clear(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._initialized = False
