@echo off
REM ============================================
REM  Qwen3.5-9B Legal — llama.cpp Server (Windows)
REM ============================================
REM  Prerequisites:
REM    1. Build llama.cpp with CUDA:
REM       git clone https://github.com/ggml-org/llama.cpp
REM       cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
REM       cmake --build llama.cpp/build --config Release -j --target llama-server
REM
REM    2. Copy llama-server.exe to this folder, OR update the path below
REM
REM  Usage: double-click this file or run: run_server.bat
REM ============================================

SET MODEL=FTunedModel\Qwen3.5-9B.Q4_K_M.gguf
SET MMPROJ=FTunedModel\Qwen3.5-9B.BF16-mmproj.gguf
SET PORT=8080

REM Try local llama-server first, then check llama.cpp/build path
IF EXIST llama-server.exe (
    SET SERVER=llama-server.exe
) ELSE IF EXIST llama.cpp\build\bin\Release\llama-server.exe (
    SET SERVER=llama.cpp\build\bin\Release\llama-server.exe
) ELSE IF EXIST llama.cpp\build\bin\llama-server.exe (
    SET SERVER=llama.cpp\build\bin\llama-server.exe
) ELSE (
    echo ERROR: llama-server.exe not found!
    echo Build it first:
    echo   git clone https://github.com/ggml-org/llama.cpp
    echo   cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
    echo   cmake --build llama.cpp/build --config Release -j --target llama-server
    pause
    exit /b 1
)

echo Starting llama.cpp server...
echo   Model:  %MODEL%
echo   MMProj: %MMPROJ%
echo   Port:   %PORT%
echo   GPU:    all layers offloaded
echo.
echo Server will be available at http://localhost:%PORT%
echo Press Ctrl+C to stop
echo.

%SERVER% ^
    --model %MODEL% ^
    --mmproj %MMPROJ% ^
    --n-gpu-layers 99 ^
    --ctx-size 4096 ^
    --batch-size 512 ^
    --ubatch-size 256 ^
    --port %PORT% ^
    --flash-attn on

pause