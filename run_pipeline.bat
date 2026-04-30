@echo off
echo =========================================
echo       GOLD AI SYSTEM - MASTER RUNNER     
echo =========================================

echo.
echo [1/4] Running ICT Feature Engineering (This may take a few minutes)...
python src\features.py

echo.
echo [2/4] Training Machine Learning Model...
python src\model.py

echo.
echo [3/4] Running Historical Backtest...
python src\backtest.py

echo.
echo [4/4] Launching Live Dashboard...
streamlit run src\dashboard.py

pause
