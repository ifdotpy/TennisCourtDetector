from ultralytics import YOLO

# First create the model from YAML
model = YOLO("yolo11n-pose.pt")

results = model.export(format="engine", data="coco-pose.yaml", imgsz=640, int8=True, batch=96)

print(results)