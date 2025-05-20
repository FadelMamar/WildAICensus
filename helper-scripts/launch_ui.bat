call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

start helper-scripts\launch_mlflow_server.bat

call deactivate

call .venv\Scripts\activate

@REM call cd UI

@REM call set LABEL_STUDIO_API_KEY=
@REM call set LABEL_STUDIO_URL=http://localhost:8080

@REM call set TRAINING_API_URL = ...
@REM call set TRAINING_API_KEY = ...

call streamlit run UI/app.py

call deactivate
