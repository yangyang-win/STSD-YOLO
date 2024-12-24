from ultralytics import YOLO
model = YOLO("ultralytics/runs/detect/STSD-YOLOv9/weights/best.pt")  # 

# Use the model
if __name__ == '__main__':
    # Use the model

    results = model.val(data='tt100k.yaml', device=1)

