@echo off
REM ===========================================================================
REM  One-click launcher for the face recognition app.
REM  Double-click this file. A browser page will open automatically.
REM
REM  NOTE: keep this file pure ASCII. Chinese characters inside a .bat file
REM  break cmd.exe parsing on Chinese Windows (it reads the file as GBK,
REM  not UTF-8), which produces confusing "not recognized as a command" errors.
REM ===========================================================================

setlocal

REM --- Run from this file's folder so main.py / config.py are found ---
cd /d "%~dp0"

REM --- Pick the Python that has cv2 / face_recognition installed.
REM --- 1) Prefer whatever "python" is on PATH (works for most people).
set "PY=python"
python -c "import cv2, face_recognition" >nul 2>&1
if errorlevel 1 set "PY="

REM --- 2) Fall back to a known full path (edit this if yours differs).
if not defined PY set "PY="

if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"

if not defined PY goto nopython

"%PY%" face_app.py %*
if errorlevel 1 goto failed

goto end

:nopython
echo.
echo [ERROR] Could not find a Python with cv2 + face_recognition installed.
echo.
echo Install the dependencies first (see README.md), or edit this .bat file
echo and point PY= at your own python.exe.
echo.
pause
goto end

:failed
echo.
echo [ERROR] The app exited with an error. The message above explains why.
echo         Common causes:
echo           - camera is in use by another program (WeChat / DingTalk / Teams)
echo           - dependencies missing: run  check_setup.py  first
echo.
pause

:end
endlocal
