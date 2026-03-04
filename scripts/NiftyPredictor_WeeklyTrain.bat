@echo off
cd /d "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor"
"C:\Users\vidhi gupta\AppData\Local\Programs\Python\Python311\python.exe" -m scripts.train_models --email >> "C:\Users\vidhi gupta\Documents\Amvi\VSCode\nifty_predictor\storage\logs\train_models_scheduler.log" 2>&1
