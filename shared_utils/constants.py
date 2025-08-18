# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\shared_utils\constants.py

"""
Legal Assistant AI - Constants and Configuration
Contains all constant values, messages, and shared configuration for the project
"""

from enum import Enum
from typing import Dict, List, Tuple
import os
from pathlib import Path

# ============================================================================
# PROJECT INFORMATION
# ============================================================================

PROJECT_NAME = "Legal Assistant AI"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Claude AI Assistant"
PROJECT_DESCRIPTION = "هوشمند سازی امور حقوقی حوزه پژوهش و فناوری"

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Base project directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_PHASE_1_DIR = DATA_DIR / "processed_phase_1"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
BACKUP_DIR = BASE_DIR / "backup"

# Configuration files
CONFIG_FILE = CONFIG_DIR / "config.json"
ENV_FILE = BASE_DIR / ".env"

# ============================================================================
# DOCUMENT TYPES AND CLASSIFICATIONS
# ============================================================================

class DocumentType(Enum):
    """Legal document types in Persian legal system"""
    POLICY = "سیاست"
    LAW = "قانون"
    REGULATION = "آیین‌نامه"
    DIRECTIVE = "دستورالعمل"
    RESOLUTION = "مصوبه"
    STATUTE = "اساسنامه"
    GUIDELINE = "راهنما"
    CIRCULAR = "بخشنامه"

class ApprovalAuthority(Enum):
    """Legal approval authorities"""
    PARLIAMENT = "مجلس شورای اسلامی"
    CABINET = "هیئت وزیران"
    SUPREME_COUNCIL = "شورای عالی انقلاب فرهنگی"
    SCIENCE_COUNCIL = "شورای عالی علوم، تحقیقات و فناوری"
    MINISTRY = "وزارت علوم، تحقیقات و فناوری"
    JUDICIARY = "قوه قضاییه"
    LEADER_OFFICE = "دفتر مقام معظم رهبری"

class DocumentSection(Enum):
    """Document sections in the legal database"""
    SECTION_1 = "بخش اول - سیاست‌های کلی"
    SECTION_2 = "بخش دوم - قوانین"
    SECTION_3 = "بخش سوم - آیین‌نامه‌ها"
    SECTION_4 = "بخش چهارم - مصوبات شورای عالی"
    SECTION_5 = "بخش پنجم - شورای علوم"
    SECTION_6 = "بخش ششم - وزارت علوم"
    SECTION_7 = "بخش هفتم - قوه قضاییه"

# ============================================================================
# PERSIAN TEXT PROCESSING CONSTANTS
# ============================================================================

# Persian digits mapping
PERSIAN_DIGITS = {
    '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
    '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
}

ENGLISH_DIGITS = {v: k for k, v in PERSIAN_DIGITS.items()}

# Common Persian legal terms
LEGAL_TERMS = {
    "article": "ماده",
    "section": "بخش",
    "chapter": "فصل",
    "clause": "بند",
    "subsection": "تبصره",
    "paragraph": "فقره",
    "law": "قانون",
    "regulation": "آیین‌نامه",
    "approval": "تصویب",
    "implementation": "اجرا"
}

# Persian text cleaning patterns
PERSIAN_CLEANUP_PATTERNS = [
    r'[\u200c\u200d\u200e\u200f]+',  # Zero-width characters
    r'[\ufeff]',  # Byte order mark
    r'[\u0640]+',  # Arabic tatweel
    r'\s+',  # Multiple spaces
]

# ============================================================================
# RAG AND LLM CONFIGURATION
# ============================================================================

# Chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

# Vector search parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.75
MAX_SEARCH_RESULTS = 10
VECTOR_DIMENSION = 384

# LLM parameters
MAX_TOKENS = 4096
TEMPERATURE = 0.1
TOP_P = 0.9

# ============================================================================
# DATABASE SCHEMA CONSTANTS
# ============================================================================

# Table names
DOCUMENTS_TABLE = "documents"
ARTICLES_TABLE = "articles"
CHUNKS_TABLE = "chunks"
EMBEDDINGS_TABLE = "embeddings"
METADATA_TABLE = "metadata"

# Database field constraints
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50000
MAX_AUTHOR_LENGTH = 200

# ============================================================================
# PERSIAN MESSAGES AND RESPONSES
# ============================================================================

class Messages:
    """Persian messages for user interface"""
    
    # Success messages
    SUCCESS_PARSE = "✅ سند با موفقیت پردازش شد"
    SUCCESS_SAVE = "✅ داده‌ها با موفقیت ذخیره شدند"
    SUCCESS_SEARCH = "✅ جستجو با موفقیت انجام شد"
    SUCCESS_VALIDATION = "✅ اعتبارسنجی موفق"
    
    # Error messages
    ERROR_FILE_NOT_FOUND = "❌ فایل مورد نظر یافت نشد"
    ERROR_INVALID_FORMAT = "❌ فرمت فایل نامعتبر است"
    ERROR_PARSING_FAILED = "❌ خطا در پردازش سند"
    ERROR_DATABASE = "❌ خطا در دسترسی به پایگاه داده"
    ERROR_LLM_CONNECTION = "❌ خطا در اتصال به مدل زبانی"
    ERROR_VALIDATION_FAILED = "❌ اعتبارسنجی ناموفق"
    
    # Warning messages
    WARNING_LARGE_FILE = "⚠️ فایل بزرگ است، پردازش ممکن است طولانی باشد"
    WARNING_LOW_CONFIDENCE = "⚠️ اطمینان پاسخ پایین است"
    WARNING_NO_RESULTS = "⚠️ نتیجه‌ای یافت نشد"
    
    # Info messages
    INFO_PROCESSING = "🔄 در حال پردازش..."
    INFO_LOADING = "📥 در حال بارگذاری..."
    INFO_SAVING = "💾 در حال ذخیره..."
    INFO_SEARCHING = "🔍 در حال جستجو..."

class PromptTemplates:
    """Persian prompt templates for LLM interactions"""
    
    QA_SYSTEM_PROMPT = """
    شما یک دستیار حقوقی متخصص در حوزه پژوهش و فناوری هستید.
    وظیفه شما پاسخگویی دقیق و مستند به سوالات حقوقی بر اساس اسناد موجود است.
    
    اصول مهم:
    - پاسخ‌های دقیق و مستند ارائه دهید
    - همیشه منبع و ماده قانونی را ذکر کنید
    - در صورت عدم اطمینان، آن را اعلام کنید
    - زبان رسمی و حقوقی استفاده کنید
    """
    
    DOCUMENT_COMPARISON_PROMPT = """
    شما یک کارشناس حقوقی هستید که باید دو سند حقوقی را مقایسه کنید.
    
    موارد مقایسه:
    - تطابق و تناقضات
    - تغییرات احتمالی
    - ارجاعات مشترک
    - درجه شباهت
    
    نتیجه را به صورت ساختاریافته ارائه دهید.
    """
    
    DRAFT_GENERATION_PROMPT = """
    شما یک نویسنده ماهر اسناد حقوقی هستید.
    
    اصول نگارش:
    - رعایت ساختار استاندارد
    - استفاده از اصطلاحات صحیح
    - انسجام و پیوستگی متن
    - مطابقت با قوانین موجود
    
    پیش‌نویس را بر اساس درخواست کاربر تهیه کنید.
    """

# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

# Persian date patterns
PERSIAN_DATE_PATTERNS = [
    r'\d{2}/\d{2}/\d{4}',  # 01/01/1400
    r'\d{1,2}/\d{1,2}/\d{4}',  # 1/1/1400
    r'\d{4}/\d{2}/\d{2}',  # 1400/01/01
]

# Article number patterns
ARTICLE_PATTERNS = [
    r'ماده\s*(\d+)',
    r'ماده\s*([۰-۹]+)',
    r'Article\s*(\d+)',
]

# Legal reference patterns
LEGAL_REFERENCE_PATTERNS = [
    r'مصوب\s*\d{2}/\d{2}/\d{4}',
    r'مصوب\s*سال\s*\d{4}',
    r'ابلاغی\s*\d{2}/\d{2}/\d{4}',
]

# ============================================================================
# FILE EXTENSIONS AND FORMATS
# ============================================================================

SUPPORTED_DOCUMENT_FORMATS = {
    'pdf': ['.pdf'],
    'word': ['.doc', '.docx'],
    'text': ['.txt'],
    'json': ['.json'],
    'excel': ['.xls', '.xlsx']
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ============================================================================
# API AND NETWORKING
# ============================================================================

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RATE_LIMIT_REQUESTS = 60  # per minute

# HTTP status codes
HTTP_SUCCESS = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_section_name(section_number: int) -> str:
    """Get Persian section name by number"""
    section_map = {
        1: DocumentSection.SECTION_1.value,
        2: DocumentSection.SECTION_2.value,
        3: DocumentSection.SECTION_3.value,
        4: DocumentSection.SECTION_4.value,
        5: DocumentSection.SECTION_5.value,
        6: DocumentSection.SECTION_6.value,
        7: DocumentSection.SECTION_7.value,
    }
    return section_map.get(section_number, f"بخش {section_number}")

def persian_to_english_digits(text: str) -> str:
    """Convert Persian digits to English digits"""
    for persian, english in ENGLISH_DIGITS.items():
        text = text.replace(persian, english)
    return text

def english_to_persian_digits(text: str) -> str:
    """Convert English digits to Persian digits"""
    for english, persian in PERSIAN_DIGITS.items():
        text = text.replace(english, persian)
    return text

def validate_file_extension(filename: str, allowed_formats: List[str] = None) -> bool:
    """Validate if file extension is supported"""
    if allowed_formats is None:
        allowed_formats = [ext for exts in SUPPORTED_DOCUMENT_FORMATS.values() for ext in exts]
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_formats

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_PHASE_1_DIR,
        VECTOR_DB_DIR, BACKUP_DIR, LOGS_DIR, CONFIG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories when module is imported
create_directories()