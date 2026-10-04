import sys
from ultralytics import YOLO
name, imgsz, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = YOLO(f"weights/{name}.pt")
m.train(data="ds1/data.yaml", imgsz=imgsz, epochs=40, batch=batch, project="runs", name=f"{name}_{imgsz}",
        exist_ok=True, workers=4, verbose=False, plots=False, mosaic=1.0, patience=15)
