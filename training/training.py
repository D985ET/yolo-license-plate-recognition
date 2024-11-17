from ultralytics import YOLO
import shutil
import os

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')


os.chdir("data")
# Fine-tune the model
model.train(
    data='data.yaml',
    epochs=50,  # Number of epochs
    imgsz=640,   # Image size
    batch=16,    # Batch size
    lr0=0.01,    # Initial learning rate
    lrf=0.1,     # Final learning rate (multiplier of lr0)
    momentum=0.937,  # Momentum for SGD
    weight_decay=0.0005,  # Weight decay
    warmup_epochs=3,  # Number of warmup epochs
    augment=True,  # Enable data augmentation
    rect=True,     # Rectangular training
    save_period=10  # Save checkpoint every 10 epochs

)
#print(os.getcwd())