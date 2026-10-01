@echo off
rem Configure + build the RTX3060 (sm_86) tree.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%
cd /d "%~dp0"
cmake -B build_86 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release > build_86_configure.log 2>&1
if errorlevel 1 (echo CONFIGURE FAILED > build_86_status.txt & exit /b 1)
ninja -C build_86 -j 8 > build_86_build.log 2>&1
if errorlevel 1 (echo BUILD FAILED > build_86_status.txt & exit /b 1)
echo BUILD OK > build_86_status.txt
