# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\README.md

# 🤖 دستیار حقوقی هوشمند - Legal Assistant AI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Phase%200%20Complete-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

## 📋 معرفی پروژه

دستیار حقوقی هوشمند یک سیستم پیشرفته مبتنی بر هوش مصنوعی است که برای تسهیل دسترسی، فهم و تحلیل اسناد و مقررات حقوقی در حوزه **پژوهش و فناوری** طراحی شده است.

### 🎯 اهداف اصلی

- **🔍 پاسخگویی هوشمند** به سوالات حقوقی با ارجاع دقیق به مواد قانونی
- **📊 مقایسه اسناد حقوقی** و شناسایی تناقضات و شباهت‌ها  
- **📝 تولید پیش‌نویس** قوانین، آیین‌نامه‌ها و دستورالعمل‌ها
- **🔗 ایجاد پایگاه داده جامع** از اسناد حقوقی حوزه علوم و فناوری

### 👥 مخاطبان هدف

- مدیران فناوری وزارت علوم
- کارشناسان حقوقی مؤسسات تحقیقاتی  
- پژوهشگران و اعضای هیئت علمی
- مشاوران حقوقی

## 🛠️ تکنولوژی‌های استفاده شده

### 🧠 هوش مصنوعی
- **RAG (Retrieval-Augmented Generation)** برای پاسخگویی دقیق
- **Local LLM:** Ollama + مدل‌های فارسی (Qwen2.5, Mistral)
- **Cloud LLM:** OpenAI GPT-4, Anthropic Claude
- **Embedding:** مدل‌های چندزبانه + مدل‌های تخصصی فارسی

### 💾 پایگاه داده
- **SQLite** برای داده‌های ساختاریافته
- **ChromaDB** برای vector search
- **JSON** برای داده‌های پردازش شده

### 🌐 وب اپلیکیشن
- **Backend:** Python + FastAPI
- **Frontend:** HTML/CSS/JavaScript
- **Database ORM:** SQLAlchemy

## 📁 ساختار پروژه

```
legal_assistant_project/
├── 📂 data/
│   ├── raw/                    # فایل‌های خام اسناد حقوقی
│   ├── processed_phase_1/      # خروجی فاز یک
│   └── vector_db/              # پایگاه داده vector
├── 📂 phase_0_setup/           # تنظیمات اولیه
├── 📂 phase_1_data_processing/ # پردازش اسناد
├── 📂 shared_utils/            # ابزارهای مشترک
├── 📂 config/                  # فایل‌های تنظیمات
├── 📂 logs/                    # فایل‌های گزارش
└── 📂 tests/                   # تست‌ها
```

## 🚀 راه‌اندازی سریع

### 1️⃣ پیش‌نیازها

```bash
# نصب Python 3.11+
# نصب Miniconda/Anaconda
# نصب Git
```

### 2️⃣ ایجاد محیط توسعه

```bash
# کلون پروژه
git clone <repository-url>
cd legal_assistant_project

# ایجاد محیط conda
conda create -n claude-ai python=3.11
conda activate claude-ai

# نصب dependencies
pip install -r requirements.txt
```

### 3️⃣ تنظیم خودکار محیط

```bash
# اجرای اسکریپت تنظیم خودکار
python phase_0_setup/environment_setup.py
```

این اسکریپت موارد زیر را به صورت خودکار انجام می‌دهد:
- ✅ بررسی نسخه Python
- ✅ بررسی محیط Conda  
- ✅ نصب dependencies مفقود
- ✅ ایجاد ساختار دایرکتوری‌ها
- ✅ تنظیم configuration
- ✅ بررسی تنظیمات Ollama
- ✅ اعتبارسنجی نهایی محیط

### 4️⃣ تنظیم Ollama (اختیاری)

```bash
# نصب Ollama
# https://ollama.ai/download

# شروع سرویس Ollama  
ollama serve

# نصب مدل‌های فارسی
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b
```

### 5️⃣ تست محیط

```bash
# تست تنظیمات
python test_environment_setup.py

# تست shared utilities
python -c "from shared_utils import get_config; print('✅ محیط آماده است')"
```

## 📊 وضعیت فازهای توسعه

### ✅ فاز صفر: آماده‌سازی محیط (تکمیل شده)
- [x] ساختار پروژه
- [x] ابزارهای مشترک (logging, config, file utils)
- [x] تنظیمات اولیه
- [x] اسکریپت‌های تنظیم خودکار

### 🔄 فاز یک: پردازش داده‌ها (در حال توسعه)
- [ ] پارسر اسناد حقوقی
- [ ] استخراج متادیتا
- [ ] تفکیک مواد و تبصره‌ها
- [ ] ایجاد پایگاه داده ساختاریافته

### ⏳ فازهای آتی
- **فاز دو:** پایگاه داده و Vector Store
- **فاز سه:** سیستم RAG  
- **فاز چهار:** رابط کاربری وب
- **فاز پنج:** قابلیت‌های پیشرفته

## 🔧 تنظیمات مهم

### فایل‌های کلیدی
- `config/config.json` - تنظیمات اصلی
- `.env` - متغیرهای محیطی
- `requirements.txt` - dependencies پایتون

### مدل‌های پیش‌فرض
- **LLM اصلی:** Qwen2.5:7b-instruct
- **LLM پشتیبان:** Mistral:7b  
- **Embedding:** paraphrase-multilingual-MiniLM
- **Persian Embedding:** HooshvareLab/bert-fa-base-uncased

## 📝 نحوه استفاده

### تست سریع
```python
from shared_utils import get_logger, get_config, read_document

# دریافت logger
logger = get_logger("MyApp")
logger.info("Test message", "پیام تست")

# دریافت تنظیمات  
config = get_config()
print(f"Project: {config.project_name}")

# خواندن سند
result = read_document("path/to/document.pdf")
if result['success']:
    print(f"Content: {result['content'][:100]}...")
```

### اجرای فاز بعدی
```bash
# پس از تکمیل فاز صفر
python phase_1_data_processing/document_parser.py
```

## 📊 گزارش‌ها و مانیتورینگ

### فایل‌های لاگ
- `logs/Legal_Assistant_AI.log` - لاگ‌های عمومی
- `logs/Legal_Assistant_AI_errors.log` - فقط خطاها  
- `logs/Legal_Assistant_AI_structured.jsonl` - لاگ‌های JSON
- `logs/environment_setup_report.json` - گزارش تنظیم محیط

### مانیتورینگ عملکرد
```python
# بررسی وضعیت سیستم
from shared_utils.config_manager import get_config
config = get_config()

print(f"🗃️ Database: {config.database.url}")
print(f"🤖 LLM Model: {config.llm.ollama_model}")  
print(f"🔍 Chunk Size: {config.rag.chunk_size}")
```

## 🤝 مشارکت در توسعه

### ساختار توسعه
1. هر فاز در یک چت جداگانه توسعه می‌یابد
2. تست کامل قبل از انتقال به فاز بعدی
3. مستندسازی کامل هر مرحله
4. نگه‌داری سادگی و خوانایی کد

### استانداردهای کدنویسی
- **Python:** PEP 8 + Type Hints
- **Persian Messages:** برای کاربران نهایی
- **English Code:** متغیرها و توابع
- **Comprehensive Logging:** فارسی + انگلیسی

## 📞 پشتیبانی

### مستندات
- `docs/` - مستندات تفصیلی
- `logs/` - گزارش‌های سیستم
- `tests/` - تست‌های موجود

### عیب‌یابی رایج
1. **خطای import:** بررسی فعال بودن محیط conda
2. **خطای Ollama:** بررسی اجرای سرویس ollama serve
3. **خطای فایل:** بررسی مجوزهای دسترسی

### مراحل عیب‌یابی
```bash
# 1. بررسی محیط
conda info --envs
python --version

# 2. تست imports
python -c "from shared_utils import PROJECT_NAME; print(PROJECT_NAME)"

# 3. بررسی لاگ‌ها
cat logs/Legal_Assistant_AI_errors.log

# 4. اجرای مجدد setup
python phase_0_setup/environment_setup.py
```

---

## 📄 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است.

## 🙏 تشکر

- **OpenAI** برای GPT models
- **Anthropic** برای Claude models  
- **Ollama** برای local LLM hosting
- **HuggingFace** برای Persian language models

---

**📅 آخرین به‌روزرسانی:** 2024-01-15  
**🏷️ نسخه:** 1.0.0  