# phase_1_data_processing/__init__.py
"""
Phase 1 Data Processing Package

Modules:
- document_splitter: Detect and split multiple legal documents inside a single raw file.
- text_cleaner: Normalize and clean Persian legal text; provide ASCII/Persian digit variants.
- metadata_extractor: Extract basic metadata (title, type, authority, date, section).
- legal_structure_parser: Light parser for chapters/articles/notes/clauses.
- json_formatter: Build canonical JSON and write per-document outputs + aggregate stats.
- main_processor: Orchestrates the full Phase 1 pipeline end-to-end.
"""

# Import only existing modules to avoid import errors
try:
    from .legal_structure_parser import LegalStructureParser, ParsedStructure
except ImportError:
    pass

try:
    from .metadata_extractor import MetadataExtractor, Metadata
except ImportError:
    pass

# Comment out missing imports for now
# from .document_splitter import DocumentSplitter, SplitResult
# from .text_cleaner import TextCleaner, CleanResult
# from .json_formatter import JsonFormatter
# from .main_processor import Phase1Processor, Phase1Report

__all__ = [
    "LegalStructureParser", "ParsedStructure",
    "MetadataExtractor", "Metadata",
    # "DocumentSplitter", "SplitResult",
    # "TextCleaner", "CleanResult", 
    # "JsonFormatter",
    # "Phase1Processor", "Phase1Report",
]
