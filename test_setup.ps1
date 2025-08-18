# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_setup.ps1

# ============================================================================
# Legal Assistant AI - PowerShell Setup Test Script
# ============================================================================

Write-Host "🔍 بررسی فایل‌های ایجاد شده..." -ForegroundColor Green

# Change to project directory
Set-Location "D:\OneDrive\AI-Project\Claude-ai-assist-mvp2"

# Check if files exist
$files = @("requirements.txt", ".env.template", ".gitignore")

foreach ($file in $files) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        Write-Host "✅ $file - موجود ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "❌ $file - یافت نشد" -ForegroundColor Red
    }
}

Write-Host "`n📂 ساختار دایرکتوری:" -ForegroundColor Cyan
Get-ChildItem -Directory | ForEach-Object { 
    Write-Host "  📁 $($_.Name)" -ForegroundColor Yellow 
}

Write-Host "`n🐍 بررسی محیط Python:" -ForegroundColor Cyan
conda info --envs | Select-String "claude-ai"

Write-Host "`n🤖 بررسی مدل‌های Ollama:" -ForegroundColor Cyan
ollama list

Write-Host "`n✅ تست‌های فاز صفر تکمیل شد!" -ForegroundColor Green