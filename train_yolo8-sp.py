from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8n-sp.yaml")
model.train(data="/Users/yusangmin/Downloads/ultralytics-main/broiler_segmentation-1/data.yaml", epochs=100, imgsz=640, batch=4, optimizer = "SGD", device="mps",)