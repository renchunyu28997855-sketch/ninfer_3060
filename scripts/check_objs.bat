@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 91
cd /d E:\download\123\ninfer-5090-sm86
ninja -C build_86 %* > obj_check.log 2>&1
echo OBJ_EXIT=%ERRORLEVEL% >> obj_check.log
