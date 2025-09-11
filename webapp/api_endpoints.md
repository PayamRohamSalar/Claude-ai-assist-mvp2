# مستندات API سامانه پاسخگوی حقوقی هوشمند

## نمای کلی

این سند، نقطه‌های پایانی (endpoints) API سامانه پاسخگوی حقوقی هوشمند را توصیف می‌کند. تمام پاسخ‌های خطا به زبان فارسی ارائه می‌شوند.

## نقطه‌های پایانی (Endpoints)

### ۱. بررسی سلامت سامانه (Health Check)

#### GET `/api/health/`

بررسی سلامت پایه سامانه را انجام می‌دهد.

**پاسخ موفق (200):**

```json
{
  "status": "healthy",
  "message": "سامانه پاسخگوی حقوقی هوشمند فعال است",
  "timestamp": "2024-01-15T10:30:00",
  "version": "0.1.0"
}
```

#### GET `/api/health/detailed`

بررسی سلامت تفصیلی با وضعیت اجزای سامانه.

**پاسخ موفق (200):**

```json
{
  "status": "healthy",
  "message": "تمام اجزای سامانه در وضعیت مطلوب قرار دارند",
  "timestamp": "2024-01-15T10:30:00",
  "version": "0.1.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "rag_engine": "healthy",
    "memory": "healthy"
  }
}
```

### ۲. پرسش و پاسخ (Question & Answer)

#### POST `/api/ask`

پرسش حقوقی را دریافت کرده و پاسخ AI-powered تولید می‌کند.

**پارامترهای ورودی:**

```json
{
  "question": "سوال حقوقی به زبان فارسی (حداقل ۵ کاراکتر)",
  "top_k": 5,
  "template": "default",
  "session_id": "اختیاری - شناسه جلسه"
}
```

**پارامترها:**

- `question` (string, required): سوال حقوقی به زبان فارسی
- `top_k` (integer, optional): تعداد منابع مرتبط (۱-۲۰، پیش‌فرض: ۵)
- `template` (string, optional): نوع قالب پاسخ (پیش‌فرض: "default")
- `session_id` (string, optional): شناسه جلسه برای پیگیری مکالمات

**پاسخ موفق (200):**

```json
{
  "answer": "هیئت علمی دانشگاه عبارتند از افرادی که به صورت تمام وقت یا پاره وقت در دانشگاه مشغول تدریس و پژوهش هستند.",
  "citations": [
    {
      "document_title": "قانون دانشگاه‌ها",
      "document_uid": "law_universities_001",
      "article_number": "23",
      "note_label": "ماده 23 قانون دانشگاه‌ها"
    },
    {
      "document_title": "آیین‌نامه استخدامی هیئت علمی",
      "document_uid": "regulation_faculty_002",
      "article_number": "5",
      "note_label": "بند 5 آیین‌نامه"
    }
  ],
  "retrieved_chunks": 2,
  "processing_time": 1.23,
  "session_id": "session-uuid",
  "request_id": "request-uuid"
}
```

#### GET `/api/templates`

لیستی از قالب‌های پاسخ موجود را برمی‌گرداند.

**پاسخ موفق (200):**

```json
{
  "templates": [
    {
      "name": "default",
      "display_name": "پاسخ عادی",
      "description": "پاسخ مستقیم و کامل به سوال"
    },
    {
      "name": "compare",
      "display_name": "مقایسه",
      "description": "مقایسه بین دو متن یا مفهوم حقوقی"
    }
  ]
}
```

## نمونه درخواست‌ها (Curl Examples)

### بررسی سلامت

```bash
curl -X GET "http://localhost:8000/api/health/" \
  -H "accept: application/json"
```

### پرسش حقوقی

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "شرایط ارتقای اعضای هیئت علمی چیست؟",
    "top_k": 3,
    "template": "default"
  }'
```

### پرسش با توکن احراز هویت

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-auth-token" \
  -d '{
    "question": "قوانین استخدام دانشگاهیان",
    "top_k": 5
  }'
```

## کدهای وضعیت HTTP

- **200**: موفق - درخواست با موفقیت انجام شد
- **401**: غیرمجاز - توکن احراز هویت نامعتبر یا وجود ندارد
- **422**: داده نامعتبر - پارامترهای ورودی نامعتبر
- **500**: خطای سرور - خطای داخلی سرور
- **503**: سرویس غیرقابل دسترس - RAG engine یا vector database در دسترس نیست

## پیام‌های خطای فارسی

### خطای سرویس (503)

```json
{
  "error": "خطا در سرویس",
  "message": "سرویس پاسخگویی در حال حاضر در دسترس نیست",
  "request_id": "uuid",
  "trace_id": "uuid"
}
```

### خطای پردازش (500)

```json
{
  "error": "خطا در پردازش سوال",
  "message": "متأسفانه در حال حاضر امکان پاسخگویی وجود ندارد. لطفاً دوباره تلاش کنید.",
  "request_id": "uuid"
}
```

### خطای اعتبارسنجی (422)

```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "ensure this value has at least 5 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

## عیب‌یابی (Troubleshooting)

### مشکلات رایج و راه‌حل‌ها

#### ۱. خطای "سرویس پاسخگویی در حال حاضر در دسترس نیست" (503)

**علت‌های ممکن:**

- Vector database در دسترس نیست
- RAG engine بارگذاری نشده
- مشکلات اتصال به مدل‌های زبانی

**راه‌حل‌ها:**

```bash
# بررسی وضعیت سرویس‌ها
curl http://localhost:8000/api/health/detailed

# راه‌اندازی مجدد pipeline
python pipeline/run_phases_1_to_3.py

# بررسی لاگ‌ها
tail -f logs/rag_engine.log
```

#### ۲. خطای "خطا در پردازش سوال" (500)

**علت‌های ممکن:**

- خطای داخلی در پردازش
- مشکلات حافظه
- Timeout در پردازش

**راه‌حل‌ها:**

```bash
# بررسی فضای دیسک و حافظه
df -h
free -h

# راه‌اندازی مجدد سرور
python -m webapp.app

# بررسی لاگ‌های مفصل
tail -f webapp/rag_engine.log
```

#### ۳. خطای "ensure this value has at least 5 characters" (422)

**علت:**

- سوال ورودی خیلی کوتاه است

**راه‌حل:**

- سوال را حداقل به ۵ کاراکتر افزایش دهید
- از سوالات کامل و واضح استفاده کنید

#### ۴. خطای اتصال (Connection Error)

**علت‌های ممکن:**

- سرور در حال اجرا نیست
- پورت اشتباه
- مشکلات شبکه

**راه‌حل‌ها:**

```bash
# بررسی وضعیت سرور
ps aux | grep uvicorn

# راه‌اندازی سرور
python -m webapp.app

# بررسی پورت
netstat -tlnp | grep 8000
```

### لاگ‌های مهم

لاگ‌های زیر را برای عیب‌یابی بررسی کنید:

- `logs/rag_engine.log` - لاگ‌های RAG engine
- `webapp/rag_engine.log` - لاگ‌های وب‌اپلیکیشن
- `embedding_generation.log` - لاگ‌های تولید embedding

### تنظیمات مهم

برای تنظیمات پیشرفته، فایل‌های زیر را بررسی کنید:

- `config/config.json` - تنظیمات کلی
- `webapp/core/config.py` - تنظیمات FastAPI
- `phase_4_llm_rag/Rag_config.json` - تنظیمات RAG engine

## نکات مهم

1. **تمام پاسخ‌ها به زبان فارسی** ارائه می‌شوند
2. **حداکثر طول سوال**: ۱۰۰۰ کاراکتر
3. **محدوده top_k**: ۱ تا ۲۰
4. **قالب‌های پشتیبانی شده**: default, compare, draft
5. **احراز هویت**: از طریق HTTP Bearer token (اختیاری)

## تماس و پشتیبانی

در صورت بروز مشکل یا نیاز به راهنمایی بیشتر، لطفاً با تیم توسعه تماس بگیرید.
