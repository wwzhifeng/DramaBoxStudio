@echo off
cd /d "%~dp0"

REM Embedded Python 3.12 environment
SET "PYTHON_PATH=%cd%\WZF312"
SET "PYTHONHOME="
SET "PYTHONPATH="
SET "PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe"
SET "PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe"
SET "PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%"
SET "PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages"

REM CUDA / cuDNN paths (from torch pip package)
SET "CUDA_HOME=%PYTHON_PATH%\Lib\site-packages\torch"

REM Offline mode - no HF network
SET "HF_HOME="
SET "HF_ENDPOINT="
SET "HF_HUB_OFFLINE=1"
SET "TRANSFORMERS_OFFLINE=1"

REM Bypass system proxy for localhost (Gradio health check)
SET "NO_PROXY=127.0.0.1,localhost"
SET "no_proxy=127.0.0.1,localhost"

REM Set PATH - embedded Python first (overrides system Python)
SET "PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts%;%PATH%"

if not exist "%PYTHON_EXECUTABLE%" (
    echo [ERROR] Embedded Python not found: %PYTHON_EXECUTABLE%
    echo Please check that WZF312 directory exists.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   DramaBox - Expressive TTS Voice Clone
echo   Loading models, please wait...
echo ========================================
echo.
echo Python: %PYTHON_EXECUTABLE%
echo.

"%PYTHON_EXECUTABLE%" -s app.py
pause
