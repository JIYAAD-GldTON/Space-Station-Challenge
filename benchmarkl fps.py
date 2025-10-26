from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/Jiyaad/Projects/Hackathon2_scripts/own/runs/detect/train5/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)        # request 60fps camera if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, imgsz=320, conf=0.5, device="cpu", stream=True, half=True)
    for result in results:  # Iterate through the generator
        annotated = result.plot()
        break  # Only process first resultk

cap.release()
cv2.destroyAllWindows()