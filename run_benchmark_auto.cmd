@echo off
setlocal
echo =======================================
echo DeepParse Smart Benchmark Runner
echo =======================================
echo.
echo [IMPORTANT] This script auto-detects corpus state.
echo [IMPORTANT] It will skip rebuild if the corpus is already ready.
echo [IMPORTANT] Wait until you see "COMPLETED!" or "FAILED!"
echo.
pause

cd /d "%~dp0"

echo.
echo Starting... Please wait.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_benchmark_auto.ps1"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
if %EXIT_CODE% neq 0 (
echo =======================================
echo FAILED! Press any key to exit.
echo =======================================
) else (
echo =======================================
echo COMPLETED! Press any key to exit.
echo =======================================
)
pause >nul

endlocal
