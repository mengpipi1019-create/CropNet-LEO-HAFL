@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_mmst_compare.ps1" -Action Pause
echo Wait for PAUSED in the training window before closing it.
pause
