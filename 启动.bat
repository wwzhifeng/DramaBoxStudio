@echo off
cd /d "%~dp0"

SET "PYTHON_PATH=%cd%\WZF312"
SET "PYTHONHOME="
SET "PYTHONPATH="
SET "PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe"
SET "PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe"
SET "PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%"
SET "PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages"

SET "CUDA_HOME=%PYTHON_PATH%\Lib\site-packages\torch"

SET "HF_HOME="
SET "HF_ENDPOINT="
SET "HF_HUB_OFFLINE=1"
SET "TRANSFORMERS_OFFLINE=1"

SET "PYTHONWARNINGS=ignore"
SET "TQDM_DISABLE=1"

SET "NO_PROXY=127.0.0.1,localhost"
SET "no_proxy=127.0.0.1,localhost"

SET "PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts%;%PATH%"

if not exist "%PYTHON_EXECUTABLE%" (
    echo [ERROR] Embedded Python not found: %PYTHON_EXECUTABLE%
    echo Please check that WZF312 directory exists.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   DramaBox Studio - AI Voice Studio
echo   Loading models, please wait...
echo ========================================
echo.
echo Python: %PYTHON_EXECUTABLE%
echo.

"%PYTHON_EXECUTABLE%" -s app.py
pause
