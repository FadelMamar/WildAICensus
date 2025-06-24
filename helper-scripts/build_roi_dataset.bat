call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_env.bat

call python tools/cli.py create_classification_data --yaml_path configs\yolo_configs\data\data_config.yaml --save_dir .tmp\cls-features --strategies "[gt, hn, fp]"^
     --detection_model_path "base_models_weights\best.pt" --roi_classifier_path "base_models_weights\roi_classifier.ckpt" --roi_cls_label_map "{0:'gt',1:'tn'}"

@REM call helper-scripts\cli.bat create_classification_data --yaml_path configs\yolo_configs\data\data_config.yaml .tmp\cls-features "[gt, hn, fp]" ^
@REM      "demo" "base_models_weights\best.pt" "base_models_weights\roi_classifier.ckpt" "{0:'gt',1:'tn'}"
