call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

start helper-scripts\launch_mlflow_server.bat

call deactivate

call .venv\Scripts\activate

@REM call cd UI

call set LABEL_STUDIO_API_KEY=4f3c25bad9334596c5b2c3b270a2d3105c8b5d4a
call set LABEL_STUDIO_URL=http://localhost:8080

@REM call set TRAINING_API_URL = ...
@REM call set TRAINING_API_KEY = ...

call streamlit run UI/app.py

call deactivate
