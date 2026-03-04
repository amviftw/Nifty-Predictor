@echo off
cd /d "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor"
"C:\Users\vidhi gupta\AppData\Local\Programs\Python\Python311\python.exe" -m scripts.evaluate_signals --days 7 --email >> "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor\storage\logs\evaluate_signals_scheduler.log" 2>&1
