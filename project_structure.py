# path: legal_assistant_project/create_project_structure.py

"""
Project structure creation script for Legal Assistant AI
Creates the complete directory structure for the project
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """
    Create the complete directory structure for the legal assistant project
    """
    
    # Base project directory
    base_dir = Path("D:\OneDrive\AI-Project\Claude-ai-assist-mvp2")
    
    # Directory structure definition
    directories = [
        # Main project directories
        "phase_0_setup",
        "phase_1_data_processing", 
        "shared_utils",
        "config",
        "tests",
        "docs",
        
        # Data directories
        "data/raw",
        "data/processed_phase_1",  # فاز صفر خروجی خاصی ندارد
        
        # Output directories for each phase
        "phase_1_data_processing/output",
        "phase_1_data_processing/logs",
        
        # Additional utility directories
        "logs",
        "backup"
    ]
    
    print("🚀 شروع ایجاد ساختار پروژه...")
    
    try:
        # Create base directory
        base_dir.mkdir(exist_ok=True)
        print(f"✅ دایرکتوری اصلی ایجاد شد: {base_dir}")
        
        # Create all subdirectories
        for directory in directories:
            dir_path = base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 {directory}")
        
        # Create __init__.py files for Python packages
        python_packages = [
            "shared_utils",
            "phase_1_data_processing"
        ]
        
        for package in python_packages:
            init_file = base_dir / package / "__init__.py"
            init_file.touch()
            print(f"🐍 {package}/__init__.py")
        
        # Create placeholder files for important directories
        placeholder_files = [
            ("data/raw", "README.md", "# فایل‌های خام اسناد حقوقی\n\nفایل‌های PDF/Word/Text اسناد حقوقی را در این دایرکتوری قرار دهید."),
            ("data/processed_phase_1", "README.md", "# خروجی فاز یک\n\nفایل‌های JSON پردازش شده اسناد حقوقی در این دایرکتوری ذخیره می‌شوند."),
            ("logs", "README.md", "# فایل‌های Log\n\nفایل‌های گزارش سیستم در این دایرکتوری ذخیره می‌شوند."),
            ("docs", "README.md", "# مستندات پروژه\n\nمستندات فنی و راهنماهای کاربری در این دایرکتوری قرار دارند.")
        ]
        
        for directory, filename, content in placeholder_files:
            file_path = base_dir / directory / filename
            file_path.write_text(content, encoding='utf-8')
            print(f"📄 {directory}/{filename}")
        
        print("\n🎉 ساختار پروژه با موفقیت ایجاد شد!")
        print(f"📍 مسیر پروژه: {base_dir.absolute()}")
        
        # Display the created structure
        print("\n📂 ساختار ایجاد شده:")
        display_tree(base_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در ایجاد ساختار: {str(e)}")
        return False


def display_tree(directory, prefix="", max_depth=3, current_depth=0):
    """
    Display directory tree structure
    """
    if current_depth >= max_depth:
        return
        
    items = sorted(directory.iterdir())
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            display_tree(item, next_prefix, max_depth, current_depth + 1)


def main():
    """Main execution function"""
    print("=" * 60)
    print("🏗️  Legal Assistant Project - Structure Creator")
    print("=" * 60)
    
    success = create_directory_structure()
    
    if success:
        print("\n✅ مرحله بعد: ایجاد محیط مجازی با miniconda")
        print("📝 دستورات پیشنهادی:")
        print("   cd legal_assistant_project")
        print("   conda create -n legal_assistant python=3.11")
        print("   conda activate legal_assistant")
    else:
        print("\n❌ خطا در ایجاد ساختار. لطفاً مجدداً تلاش کنید.")
        sys.exit(1)


if __name__ == "__main__":
    main()