@echo off
rem Starts OpenAlgo Atlas and opens it in the browser. Closing this window stops OpenAlgo.
setlocal
cd /d "%~dp0..\.."
title OpenAlgo Atlas - keep this window open (closing it stops OpenAlgo)

powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 5000); $c.Close(); exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo OpenAlgo is already running. Opening it in the browser.
    start "" http://127.0.0.1:5000
    exit /b 0
)

echo Starting OpenAlgo Atlas. Your browser opens in a few seconds.
echo Keep this window open while you use OpenAlgo; the Atlas updates itself at 7:30 pm on weekdays.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 20; Start-Process 'http://127.0.0.1:5000'"
"tools\uv.exe" run app.py
echo.
echo OpenAlgo has stopped.
pause
