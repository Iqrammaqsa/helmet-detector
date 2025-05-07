import torch
import cv2
import numpy as np
import time
import uuid
from collections import defaultdict

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/helmet-detector/weights/best.pt', source='github')

# cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1")
cap = cv2.VideoCapture(0)

# Tracking status orang
no_helmet_timers = defaultdict(lambda: {"start": None, "captured": False})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    persons = []
    helmets = []

    for *xyxy, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]
        x1, y1, x2, y2 = map(int, xyxy)

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "hat":
            helmets.append((x1, y1, x2, y2))

    # Deteksi apakah tiap orang memakai helm
    for i, (x1, y1, x2, y2) in enumerate(persons):
        center_x = (x1 + x2) // 2
        head_y = y1 + (y2 - y1) // 4

        wears_helmet = any(hx1 < center_x < hx2 and hy1 < head_y < hy2 for hx1, hy1, hx2, hy2 in helmets)

        if wears_helmet:
            no_helmet_timers[i]["start"] = None
            no_helmet_timers[i]["captured"] = False
        else:
            if no_helmet_timers[i]["start"] is None:
                no_helmet_timers[i]["start"] = time.time()
            elif not no_helmet_timers[i]["captured"] and time.time() - no_helmet_timers[i]["start"] > 5:
                # Ambil foto
                filename = f"violation_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[ALERT] Pelanggaran: orang ke-{i} tidak pakai helm > 5 detik! Foto disimpan: {filename}")
                no_helmet_timers[i]["captured"] = True

    annotated_frame = np.squeeze(results.render())
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
