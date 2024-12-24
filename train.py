from ultralytics import YOLO

model = YOLO("STSD-YOLOv9.yaml").load('runs/detect/STSD-YOLOv9/weights/best.pt') 
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='tt100k.yaml', epochs=300, batch=14, device=1, name="STSD-YOLOv9")
