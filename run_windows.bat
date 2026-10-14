@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed or not on PATH. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Starting InboxGuard at http://127.0.0.1:5000  (press Ctrl+C to stop)
start "" http://127.0.0.1:5000
python app.py
pause
