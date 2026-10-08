@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_mmst_compare.ps1" -Action Run
echo.
echo See the message above for completion, pause, or errors. Same command resumes.
pause
