"""Reports endpoints. Return JSON artefacts produced by training / eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from apps.api.dependencies import get_current_user
from core.errors import NotFoundError

router = APIRouter(prefix="/reports", tags=["reports"],
                   dependencies=[Depends(get_current_user)])


def _list_report_files() -> list[dict]:
    out: list[dict] = []
    for folder in (Path("reports/metrics"), Path("reports/monitoring"),
                   Path("reports/exports")):
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*")):
            if p.is_file():
                out.append({
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "modified_at": p.stat().st_mtime,
                })
    return out


@router.get("")
def list_reports() -> list[dict]:
    return _list_report_files()


@router.get("/{report_name}")
def get_report(report_name: str) -> dict:
    """Return JSON report contents (or a small wrapper for CSV)."""
    for folder in (Path("reports/metrics"), Path("reports/monitoring"),
                   Path("reports/exports")):
        candidate = folder / report_name
        if candidate.exists() and candidate.is_file():
            if candidate.suffix == ".json":
                return json.loads(candidate.read_text())
            if candidate.suffix == ".csv":
                return {"type": "csv", "content": candidate.read_text()}
            raise NotFoundError(f"Unsupported report type: {candidate.suffix}")
    raise NotFoundError(f"Report {report_name} not found")


@router.get("/download/{report_name}")
def download_report(report_name: str):
    for folder in (Path("reports/metrics"), Path("reports/monitoring"),
                   Path("reports/exports"), Path("reports/plots")):
        candidate = folder / report_name
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate), filename=report_name)
    raise NotFoundError(f"Report {report_name} not found")