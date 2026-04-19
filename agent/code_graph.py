import ast
import os
import sqlite3
from typing import Dict, List, Any, Tuple

from config import CODE_GRAPH_DB_PATH


class CodeGraphIndex:
    def __init__(self, db_path: str = CODE_GRAPH_DB_PATH):
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
                CREATE TABLE IF NOT EXISTS graph_files(
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_symbols(
                    path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    lineno INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_imports(
                    path TEXT NOT NULL,
                    module TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_calls(
                    path TEXT NOT NULL,
                    callee TEXT NOT NULL,
                    lineno INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_symbols_name ON graph_symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_imports_module ON graph_imports(module)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_calls_callee ON graph_calls(callee)")
            conn.commit()
            self._initialized = True
        finally:
            conn.close()

    def _needs_refresh(self, conn: sqlite3.Connection, path: str, mtime: float) -> bool:
        row = conn.execute("SELECT mtime FROM graph_files WHERE path = ?", (path,)).fetchone()
        if not row:
            return True
        return float(row["mtime"]) != float(mtime)

    def _extract_graph(self, content: str) -> Tuple[List[Tuple[str, str, int]], List[str], List[Tuple[str, int]]]:
        symbols: List[Tuple[str, str, int]] = []
        imports: List[str] = []
        calls: List[Tuple[str, int]] = []
        try:
            tree = ast.parse(content)
        except Exception:
            return symbols, imports, calls

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols.append((node.name.lower(), "function", int(node.lineno)))
            elif isinstance(node, ast.AsyncFunctionDef):
                symbols.append((node.name.lower(), "async_function", int(node.lineno)))
            elif isinstance(node, ast.ClassDef):
                symbols.append((node.name.lower(), "class", int(node.lineno)))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.lower())
            elif isinstance(node, ast.Call):
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if callee:
                    calls.append((callee.lower(), int(getattr(node, "lineno", 0) or 0)))
        return symbols, imports, calls

    def upsert_python_files(self, file_payloads: List[Dict[str, Any]]) -> int:
        if not file_payloads:
            return 0
        self.ensure_initialized()
        conn = self._connect()
        refreshed = 0
        try:
            for item in file_payloads:
                path = item["path"]
                mtime = float(item["mtime"])
                content = item["content"]
                if not self._needs_refresh(conn, path, mtime):
                    continue
                refreshed += 1

                symbols, imports, calls = self._extract_graph(content)
                conn.execute("DELETE FROM graph_symbols WHERE path = ?", (path,))
                conn.execute("DELETE FROM graph_imports WHERE path = ?", (path,))
                conn.execute("DELETE FROM graph_calls WHERE path = ?", (path,))

                conn.execute(
                    """
                    INSERT INTO graph_files(path, mtime) VALUES(?, ?)
                    ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime
                    """,
                    (path, mtime),
                )
                conn.executemany(
                    "INSERT INTO graph_symbols(path, name, kind, lineno) VALUES(?, ?, ?, ?)",
                    [(path, name, kind, lineno) for name, kind, lineno in symbols],
                )
                conn.executemany(
                    "INSERT INTO graph_imports(path, module) VALUES(?, ?)",
                    [(path, module) for module in imports],
                )
                conn.executemany(
                    "INSERT INTO graph_calls(path, callee, lineno) VALUES(?, ?, ?)",
                    [(path, callee, lineno) for callee, lineno in calls],
                )
            conn.commit()
            return refreshed
        finally:
            conn.close()

    def query_file_boosts(self, query_terms: List[str], limit: int = 200) -> Dict[str, float]:
        self.ensure_initialized()
        terms = [t.lower().strip() for t in query_terms if t.strip()]
        if not terms:
            return {}
        conn = self._connect()
        try:
            boosts: Dict[str, float] = {}
            for term in terms:
                symbol_rows = conn.execute(
                    "SELECT path, COUNT(*) AS cnt FROM graph_symbols WHERE name = ? GROUP BY path ORDER BY cnt DESC LIMIT ?",
                    (term, limit),
                ).fetchall()
                for row in symbol_rows:
                    boosts[row["path"]] = boosts.get(row["path"], 0.0) + (float(row["cnt"]) * 5.0)

                call_rows = conn.execute(
                    "SELECT path, COUNT(*) AS cnt FROM graph_calls WHERE callee = ? GROUP BY path ORDER BY cnt DESC LIMIT ?",
                    (term, limit),
                ).fetchall()
                for row in call_rows:
                    boosts[row["path"]] = boosts.get(row["path"], 0.0) + (float(row["cnt"]) * 2.0)

                import_rows = conn.execute(
                    "SELECT path, COUNT(*) AS cnt FROM graph_imports WHERE module LIKE ? GROUP BY path ORDER BY cnt DESC LIMIT ?",
                    (f"%{term}%", limit),
                ).fetchall()
                for row in import_rows:
                    boosts[row["path"]] = boosts.get(row["path"], 0.0) + (float(row["cnt"]) * 1.5)

            return boosts
        finally:
            conn.close()
