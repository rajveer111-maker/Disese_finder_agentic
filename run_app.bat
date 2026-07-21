@echo off
echo Starting Agentic Disease Finder...
call agentic_env\Scripts\activate.bat
"agentic_env\Scripts\python.exe" -m streamlit run app.py
