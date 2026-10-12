@echo off
echo ============================================================
echo SatQuery AI -- SIH26167 Demo Launcher
echo ============================================================

:: Always run from the directory that contains this bat file (the project root)
cd /d "%~dp0"

:: Start the FastAPI backend on port 7860
:: PYTHONPATH set so 'satquery' package is found regardless of install state
start "SatQuery AI Backend" cmd /k "set PYTHONPATH=%~dp0 && python -m uvicorn satquery.api:app --port 7860 --reload"

:: Wait for the server to start
timeout /t 4 /nobreak > nul

:: Open the frontend (index.html lives in docs\)
start "" "%~dp0docs\index.html"

echo.
echo Backend: http://localhost:7860
echo API docs: http://localhost:7860/docs
echo Health:   http://localhost:7860/health
echo Sites:    http://localhost:7860/sites
echo.
echo The globe will open in your browser.
echo Open a mission site, go to Ask AI tab, and run a query.
echo.
pause
