@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 91
set CC=cl
set CXX=cl
cmake -B _cfg_test -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release > _cfg_test.log 2>&1
if errorlevel 1 exit /b 1
cmake --build _cfg_test --target ninfer_media_decode >> _cfg_test.log 2>&1
echo PROBE_EXIT=%ERRORLEVEL% >> _cfg_test.log
