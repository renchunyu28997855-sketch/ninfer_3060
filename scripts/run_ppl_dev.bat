@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >/dev/null 2>&1
cd /d E:\download\123\ninfer-5090-sm86\build_120a\apps
ninfer-perplexity.exe "E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3.ninfer" --corpus "E:\download\123\ninfer-5090-sm86\eval\corpora\perplexity-1m\manifest.json" --quick --log-level info > ppl_out.txt 2>ppl_err.txt
echo PPL_EXIT=%ERRORLEVEL%
