@echo off
rem Downloads the latest OpenAlgo Atlas. Your settings, account, broker keys and Atlas data are kept.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
