@echo off
REM Double-click to start the RAG Chatbot in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up Python environment...
    python -m venv .venv || (echo Python 3.10+ is required. & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
