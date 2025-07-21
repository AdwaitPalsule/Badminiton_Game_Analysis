import ultralytics 
from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="input_videos/test.mp4",save = True )