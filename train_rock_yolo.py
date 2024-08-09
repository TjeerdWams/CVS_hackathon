import os
import shutil
from ultralytics import YOLO

scripts_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(scripts_dir, 'data/data.yaml')

# Clean up the output directories
shutil.rmtree(os.path.join(scripts_dir, 'rock_training'), ignore_errors=True)

model = YOLO('yolov10n.pt')  # Load a pretrained model
model.train(data=yaml_path,  
            epochs=100, imgsz=640, 
            batch=16, 
            name='rock',
            project='rock_training')
