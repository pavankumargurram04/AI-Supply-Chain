"""Quality report dataclass and export helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class QualityReport:
    input_dir: str
    generated_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    row_counts: Dict[str, int] = field(default_factory=dict)
    schema_issues: List[Dict[str, Any]] = field(default_factory=list)
    range_issues: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_reports: List[Dict[str, Any]] = field(default_factory=list)
    missingness: List[Dict[str, Any]] = field(default_factory=list)
    clean_summaries: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # A dataset "passes" if there are no schema issues and no duplicate reports
        # with duplicate rows. Range issues are warnings (real data can be noisy).
        if self.schema_issues:
            return False
        for rep in self.duplicate_reports:
            if rep.get("duplicate_rows", 0) > 0:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload

    def write(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p