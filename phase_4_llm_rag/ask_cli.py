#!/usr/bin/env python3
"""
Persian Legal Question CLI

A command-line interface for asking Persian legal questions against 
precomputed RAG artifacts using the Legal RAG Engine.

Features:
- Persian question input with multiple templates
- Configurable retrieval and filtering options  
- JSON or human-readable output formats
- Source citation display
- Comprehensive error handling
"""

from __future__ import annotations
import argparse
import json
import sys
import os
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from phase_4_llm_rag.rag_engine import LegalRAGEngine


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with Persian help messages."""
    parser = argparse.ArgumentParser(
        description="Ask a Persian legal question via RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
استفاده:
  %(prog)s --question "شرایط مرخصی اعضای هیئت علمی چیست؟" --show-sources
  %(prog)s --question "تفاوت آیین‌نامه ارتقا و مصوبه X چیست؟" --template compare --top-k 7
  %(prog)s --question "برای تأسیس شرکت دانش‌بنیان چه مقرراتی لازم است؟" --filter document_type=law --pretty

قالب‌های پرسش:
  default  - پرسش عادی با پاسخ مستقیم
  compare  - مقایسه بین دو متن حقوقی
  draft    - تهیه پیش‌نویس متن حقوقی

فیلترهای قابل استفاده:
  document_type=law/regulation    - نوع سند
  section="بخش ۱"                - بخش خاص
  document_uid=specific_doc       - سند مشخص
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--question", 
        required=True, 
        help="متن پرسش به زبان فارسی (اجباری)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        default="phase_4_llm_rag/Rag_config.json",
        help="مسیر فایل تنظیمات (پیش‌فرض: phase_4_llm_rag/Rag_config.json)"
    )
    
    # Query parameters
    parser.add_argument(
        "--template", 
        default="default", 
        choices=["default", "compare", "draft"],
        help="قالب پرسش: default (عادی), compare (مقایسه), draft (پیش‌نویس)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=None,
        help="تعداد اسناد بازیابی‌شده (اختیاری، جایگزین تنظیمات پیش‌فرض)"
    )
    
    parser.add_argument(
        "--filter", 
        action="append", 
        default=[], 
        help="فیلتر به صورت key=value (قابل تکرار، مثال: --filter document_type=law)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None,
        help="حداکثر زمان انتظار برای پاسخ LLM به ثانیه"
    )
    
    # Output format options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json", 
        action="store_true", 
        help="نمایش خروجی کامل به صورت JSON"
    )
    
    output_group.add_argument(
        "--pretty", 
        action="store_true", 
        help="نمایش پاسخ به صورت زیبا و قابل خواندن"
    )
    
    parser.add_argument(
        "--show-sources", 
        action="store_true", 
        help="نمایش فهرست فشرده منابع در انتهای پاسخ"
    )
    
    return parser.parse_args()


def parse_filters(filters_kv: List[str]) -> Dict[str, str]:
    """Parse filter key=value pairs into a dictionary.
    
    Args:
        filters_kv: List of "key=value" strings
        
    Returns:
        Dictionary of parsed filters
    """
    result = {}
    for kv in filters_kv:
        if "=" not in kv:
            print(f"هشدار: فیلتر نامعتبر نادیده گرفته شد: {kv}", file=sys.stderr)
            continue
        key, value = kv.split("=", 1)
        result[key.strip()] = value.strip().strip('"\'')  # Remove quotes
    return result


def format_sources(citations: List[Dict[str, Any]]) -> str:
    """Format citations as a readable sources list.
    
    Args:
        citations: List of citation dictionaries
        
    Returns:
        Formatted sources string
    """
    if not citations:
        return ""
    
    sources_lines = ["\nمنابع:"]
    for i, citation in enumerate(citations, 1):
        # Extract citation information with fallbacks
        title = (citation.get("document_title") or 
                citation.get("document_uid") or 
                "سند نامشخص")
        
        article_num = citation.get("article_number", "نامشخص")
        note_label = citation.get("note_label", "")
        
        # Format citation line
        source_line = f"[{i}] {title} — ماده {article_num}"
        if note_label:
            source_line += f" ({note_label})"
        
        sources_lines.append(source_line)
    
    return "\n".join(sources_lines)


def print_pretty_output(result: Dict[str, Any], show_sources: bool = False) -> None:
    """Print formatted output for human readability.
    
    Args:
        result: RAG engine result dictionary
        show_sources: Whether to show sources list
    """
    answer = result.get("answer", "پاسخی دریافت نشد.")
    
    # Print main answer
    print("📋 پاسخ:")
    print("=" * 50)
    print(answer)
    
    # Print metadata
    retrieved_count = result.get("retrieved_chunks", 0)
    citations_count = len(result.get("citations", []))
    
    print("\n" + "=" * 50)
    print(f"📊 آمار: {retrieved_count} سند بازیابی‌شده، {citations_count} استناد")
    
    # Show sources if requested
    if show_sources:
        sources = format_sources(result.get("citations", []))
        if sources:
            print(sources)


def apply_runtime_overrides(engine: LegalRAGEngine, args: argparse.Namespace) -> None:
    """Apply runtime configuration overrides to the engine.
    
    Args:
        engine: The RAG engine instance
        args: Parsed command line arguments
    """
    # Override LLM timeout if specified and possible
    if args.timeout and hasattr(engine, "llm_client") and engine.llm_client:
        try:
            if hasattr(engine.llm_client, "timeout"):
                engine.llm_client.timeout = args.timeout
                print(f"⏱️  زمان انتظار LLM تنظیم شد: {args.timeout} ثانیه", file=sys.stderr)
        except Exception as e:
            print(f"هشدار: تنظیم timeout LLM ناموفق: {e}", file=sys.stderr)


def validate_config_and_artifacts(config_path: str) -> None:
    """Validate that required configuration and artifacts exist.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"فایل تنظیمات یافت نشد: {config_path}")
    
    # Load config to check artifact paths
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check critical paths mentioned in config
        critical_paths = [
            config.get("database_path"),
            config.get("chunks_file")
        ]
        
        for path in critical_paths:
            if path and not os.path.exists(path):
                print(f"هشدار: فایل اساسی یافت نشد: {path}", file=sys.stderr)
                
    except Exception as e:
        print(f"هشدار: بررسی فایل‌های اساسی ناموفق: {e}", file=sys.stderr)


def main() -> int:
    """Main CLI function.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate configuration and artifacts
        validate_config_and_artifacts(args.config)
        
        # Initialize RAG engine
        print("🔄 در حال بارگذاری موتور RAG...", file=sys.stderr)
        engine = LegalRAGEngine(config_path=args.config)
        
        # Apply runtime overrides
        apply_runtime_overrides(engine, args)
        
        # Parse filters
        filters = parse_filters(args.filter)
        if filters:
            print(f"🔍 فیلترهای اعمال‌شده: {filters}", file=sys.stderr)
        
        # Determine top_k value
        top_k = args.top_k
        if top_k is None:
            # Use default from config if available
            try:
                top_k = engine.config.get("retriever", {}).get("top_k", 5)
            except:
                top_k = 5
        
        print(f"🔍 در حال پردازش پرسش با {top_k} سند برتر...", file=sys.stderr)
        
        # Get answer from RAG engine
        result = engine.answer(
            question=args.question,
            top_k=top_k,
            template_name=args.template,
            filters=filters if filters else None
        )
        
        print("✅ پردازش کامل شد.", file=sys.stderr)
        
        # Output results based on format choice
        if args.json:
            # JSON output
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.pretty:
            # Pretty formatted output
            print_pretty_output(result, show_sources=args.show_sources)
        else:
            # Plain answer output
            answer = result.get("answer", "پاسخی دریافت نشد.")
            print(answer)
            
            # Show sources if requested
            if args.show_sources:
                sources = format_sources(result.get("citations", []))
                if sources:
                    print(sources)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ خطا: فایل یا مسیر مورد نیاز پیدا نشد.", file=sys.stderr)
        print(f"جزئیات: {e}", file=sys.stderr)
        print("\nلطفاً بررسی کنید که فایل‌های Phase 2 و Phase 3 موجود باشند.", file=sys.stderr)
        return 2
        
    except KeyboardInterrupt:
        print("\n⏹️  عملیات توسط کاربر متوقف شد.", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"❌ خطای غیرمنتظره در اجرای RAG: {e}", file=sys.stderr)
        print("\nبرای اطلاعات بیشتر، log فایل‌ها را بررسی کنید.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())