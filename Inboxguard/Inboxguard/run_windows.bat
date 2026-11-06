@echo off
REM One-click start on Windows: creates a virtual environment, installs dependencies, runs the app.
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements.txt
python serve.py
