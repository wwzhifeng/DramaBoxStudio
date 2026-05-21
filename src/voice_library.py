"""音色库管理 — 上传/保存/删除/列表。"""
import json
import logging
import shutil
import time
from pathlib import Path

from config import VOICES_DIR, VOICES_JSON

log = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _load_db() -> list[dict]:
    _ensure_dirs()
    if VOICES_JSON.exists():
        try:
            return json.loads(VOICES_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_db(db: list[dict]) -> None:
    _ensure_dirs()
    VOICES_JSON.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def list_voices() -> list[dict]:
    """返回 [{name, path, duration_seconds, created_at}]。"""
    db = _load_db()
    # 清理文件已被手动删除的记录
    valid = []
    changed = False
    for v in db:
        if Path(v["path"]).exists():
            valid.append(v)
        else:
            changed = True
    if changed:
        _save_db(valid)
    return valid


def save_voice(name: str, source_path: str) -> dict | None:
    """保存音色。返回 voice dict 或 None。"""
    if not name or not name.strip():
        return None
    name = name.strip()
    src = Path(source_path)
    if not src.exists():
        return None
    _ensure_dirs()
    ext = src.suffix or ".wav"
    dest = VOICES_DIR / f"{name}{ext}"
    shutil.copy2(src, dest)

    try:
        import torchaudio
        info = torchaudio.info(str(dest))
        duration = info.num_frames / info.sample_rate
    except Exception:
        duration = 0.0

    db = _load_db()
    # 同名覆盖
    db = [v for v in db if v["name"] != name]
    entry = {
        "name": name,
        "path": str(dest),
        "duration_seconds": round(duration, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    db.append(entry)
    # 按创建时间排序
    db.sort(key=lambda v: v.get("created_at", ""))
    _save_db(db)
    log.info(f"音色已保存: {name} ({duration:.1f}s)")
    return entry


def delete_voice(name: str) -> bool:
    """删除音色及其文件。"""
    db = _load_db()
    entry = next((v for v in db if v["name"] == name), None)
    if entry is None:
        return False
    try:
        Path(entry["path"]).unlink(missing_ok=True)
    except Exception:
        pass
    db = [v for v in db if v["name"] != name]
    _save_db(db)
    log.info(f"音色已删除: {name}")
    return True
