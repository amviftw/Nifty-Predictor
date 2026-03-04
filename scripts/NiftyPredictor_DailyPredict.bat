@echo off
cd /d "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor"
"C:\Users\vidhi gupta\AppData\Local\Programs\Python\Python311\python.exe" -m scripts.daily_predict --email >> "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor\storage\logs\daily_predict_scheduler.log" 2>&1
