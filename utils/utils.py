# utils.py

from ultralytics import YOLO
import cv2
import os

def detect_and_save(video_path, model_path="yolov8n.pt", conf_threshold=0.5, save_path="outputs/"):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video")

    os.makedirs(save_path, exist_ok=True)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        os.path.join(save_path, "annotated_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_num += 1
        print(f"[Frame {frame_num}] Detected: {len(results[0].boxes)} objects")

    cap.release()
    out.release()
    print("✅ Detection complete. Saved to:", os.path.join(save_path, "annotated_output.mp4"))
