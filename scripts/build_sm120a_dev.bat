@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 91
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set CC=cl
set CXX=cl
cmake -B build_120a -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release > build_120a.log 2>&1
if errorlevel 1 (echo CONFIGURE_FAILED >> build_120a.log & exit /b 1)
cmake --build build_120a -j 8 >> build_120a.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL% >> build_120a.log
