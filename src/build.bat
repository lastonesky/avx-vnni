@echo off
rem ---------------------------------------------------------------
rem  Build AVX-VNNI demo (MSVC x64 + MASM64)
rem  从 "x64 Native Tools Command Prompt for VS" 运行，或双击本文件
rem ---------------------------------------------------------------
setlocal
set VSDIR=C:\Program Files\Microsoft Visual Studio\18\Enterprise

if not exist "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [!] 找不到 vcvarsall.bat，请修改本文件里的 VSDIR
    goto :eof
)
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul

echo === compiling vnni_demo.c ===
cl    /nologo /O2 /arch:AVX2 /utf-8 /std:c17 /EHsc /W3 /c vnni_demo.c || goto :eof

echo === assembling vnni_asm.asm ===
ml64  /nologo /c /Cx /W3 vnni_asm.asm || goto :eof

echo === linking ===
link  /nologo /subsystem:console /machine:x64 vnni_demo.obj vnni_asm.obj /out:vnni_demo.exe || goto :eof

echo.
echo === running ===
vnni_demo.exe
echo.
echo exit code = %ERRORLEVEL%
endlocal
