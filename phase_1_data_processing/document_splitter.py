# phase_1_data_processing/document_splitter.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Optional

from shared_utils.logger import get_logger
from shared_utils.file_utils import DocumentReader
from shared_utils.constants import (
    Messages,
    # directory constants expected from Phase 0
    RAW_DATA_DIR,  # e.g., Path("data/raw")
    # if PROCESSED_PHASE_1_DIR not present in your constants,
    # you can set a fallback: Path("data/processed_phase_1")
    # but here we assume it exists in Phase 0 constants
    # similar to LOGS_DIR in logger usage
)

try:
    # Optional; if constant exists use it. Otherwise fallback.
    from shared_utils.constants import PROCESSED_PHASE_1_DIR
except Exception:
    PROCESSED_PHASE_1_DIR = Path("data/processed_phase_1")


DOC_TITLE_PATTERN = re.compile(
    r"(?m)^\s*(?:قانون|آیین[\u200c‌]?نامه|اساسنامه|دستورالعمل|مصوبه|بخشنامه)\s+[^\n]+"
)

@dataclass
class SplitResult:
    """A single detected document inside a multi-document file."""
    idx: int
    title: str
    content: str
    source_file: str
    start_char: int
    end_char: int


class DocumentSplitter:
    """
    Detect document boundaries and split a multi-document file into
    separate (title, content) chunks for downstream processing.
    """

    def __init__(self, processed_dir: Path | None = None) -> None:
        self.logger = get_logger("DocumentSplitter")
        self.processed_dir = processed_dir or PROCESSED_PHASE_1_DIR

    def split_file(self, file_path: Path) -> List[SplitResult]:
        """Split a single file into multiple documents."""
        self.logger.info(
            f"Splitting file: {file_path.name}",
            "تفکیک اسناد در فایل ورودی",
            file=str(file_path)
        )

        reader = DocumentReader()
        res = reader.read_document(file_path)
        if not res.get("success"):
            self.logger.error(
                f"Unable to read document: {res.get('error')}",
                "عدم امکان خواندن فایل ورودی",
                file=str(file_path),
                error=res.get("error")
            )
            return []

        text: str = res["content"]
        return self._split_text(text, file_path)

    def _split_text(self, text: str, file_path: Path) -> List[SplitResult]:
        matches = list(DOC_TITLE_PATTERN.finditer(text))
        results: List[SplitResult] = []

        if not matches:
            # Fallback: treat whole text as a single doc if non-empty
            trimmed = text.strip()
            if trimmed:
                results.append(
                    SplitResult(
                        idx=0,
                        title=(trimmed.split("\n", 1)[0][:120] or "سند بدون عنوان"),
                        content=trimmed,
                        source_file=file_path.name,
                        start_char=0,
                        end_char=len(text),
                    )
                )
            self._save_intermediate(file_path, results)
            return results

        # Build segments between match spans
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            title = m.group(0).strip()
            results.append(
                SplitResult(
                    idx=i,
                    title=title,
                    content=chunk,
                    source_file=file_path.name,
                    start_char=start,
                    end_char=end,
                )
            )

        self.logger.info(
            f"Detected {len(results)} document(s)",
            "تعداد اسناد شناسایی‌شده",
            count=len(results),
            source=file_path.name
        )

        self._save_intermediate(file_path, results)
        return results

    def _save_intermediate(self, file_path: Path, results: Iterable[SplitResult]) -> Path:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.processed_dir / f"split_{file_path.stem}.json"
        data = [asdict(r) for r in results]

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Intermediate split saved: {out_path.name}",
            "فایل میانی تفکیک ذخیره شد",
            output=str(out_path)
        )
        return out_path
