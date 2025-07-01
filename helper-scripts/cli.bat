@echo off

cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"
cd "D:\datalabeling"

call deactivate

call helper-scripts\activate_env.bat

call python tools/cli.py %*
