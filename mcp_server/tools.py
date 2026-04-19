import os
from typing import List, Dict, Tuple, Any, Optional
from config import ALLOWED_EXTENSIONS, MAX_FILE_BYTES, IGNORE_DIRS, MAX_SCAN_FILES

_file_cache: Dict[str, str] = {}
_mtime_cache: Dict[str, float] = {}

def list_files(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    files = []
    stats = {"skipped_ext": 0, "skipped_size": 0, "skipped_error": 0, "total_scanned": 0, "skipped_limit": 0}
    
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in IGNORE_DIRS]
        
        for f in filenames:
            stats["total_scanned"] += 1
            if len(files) >= MAX_SCAN_FILES:
                stats["skipped_limit"] += 1
                continue
                
            if f.startswith("."):
                stats["skipped_ext"] += 1
                continue
                
            ext = os.path.splitext(f)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                stats["skipped_ext"] += 1
                continue
                
            full_path = os.path.join(root, f)
            try:
                stat_res = os.stat(full_path)
                if stat_res.st_size > MAX_FILE_BYTES:
                    stats["skipped_size"] += 1
                    continue
                files.append({
                    "path": full_path,
                    "mtime": stat_res.st_mtime
                })
            except OSError:
                stats["skipped_error"] += 1
                continue
                
    files.sort(key=lambda x: x["path"])  
    return files, stats

def read_file(path: str) -> Optional[str]:
    try:
        mtime = os.path.getmtime(path)
        if path in _file_cache and _mtime_cache.get(path) == mtime:
            return _file_cache[path]
            
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            _file_cache[path] = content
            _mtime_cache[path] = mtime
            return content
    except Exception:
        return None
