@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
echo MSVC environment loaded
echo Installing packages...
".venv\Scripts\uv.exe" pip install -r requirements.txt
echo Exit code: %ERRORLEVEL%
pause
