@echo off
REM Run MuseTalk API Server

REM Set the script directory as working directory
cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the server
python -m server.main

