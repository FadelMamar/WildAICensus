call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

call python tools/cli.py create_classification_data configs\yolo_configs\data\data_config.yaml .tmp\cls-features "[gt, hn, fp]"^
     "demo" "base_models_weights\best.pt" "base_models_weights\roi_classifier.ckpt" "{0:'gt',1:'tn'}"
