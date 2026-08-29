@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    py -3.12 tools\launch.py
    goto :finished
  )
  py -3 tools\launch.py
  goto :finished
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 and enable Add Python to PATH.
  echo Then run Start Moka.bat again.
  pause
  exit /b 1
)
python tools\launch.py
:finished
if errorlevel 1 (
  echo.
  echo Startup failed. The actual error is printed above.
  pause
)
endlocal
