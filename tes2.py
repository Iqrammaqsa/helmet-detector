import torch
import cv2
import numpy as np
import time
import uuid
from collections import defaultdict

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/helmet-detector/weights/best.pt', source='github')

# cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1")
cap = cv2.VideoCapture("rtsp://service:Passw0rd!23@10.146.81.55:554/cam/realmonitor?channel=1&subtype=1")

# cap = cv2.VideoCapture(0)

no_helmet_timers = defaultdict(lambda: {"start": None, "captured": False})
person_data = {}
next_person_id = 0
iou_threshold = 0.5
inactive_timeout = 3

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
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

    matched_ids = set()
    new_person_ids = []

    for box in persons:
        best_iou = 0
        best_pid = None
        for pid, data in person_data.items():
            iou_val = iou(box, data["bbox"])
            if iou_val > best_iou and iou_val > iou_threshold and pid not in matched_ids:
                best_iou = iou_val
                best_pid = pid

        if best_pid is not None:
            person_id = best_pid
            person_data[person_id]["bbox"] = box
            person_data[person_id]["last_seen"] = current_time
        else:
            person_id = next_person_id
            person_data[person_id] = {"bbox": box, "last_seen": current_time}
            next_person_id += 1

        new_person_ids.append(person_id)
        matched_ids.add(person_id)

    # Hapus ID tidak aktif
    to_delete = [pid for pid, data in person_data.items() if current_time - data["last_seen"] > inactive_timeout]
    for pid in to_delete:
        person_data.pop(pid, None)
        no_helmet_timers.pop(pid, None)

    annotated_frame = np.squeeze(results.render())

    for pid in new_person_ids:
        x1, y1, x2, y2 = person_data[pid]["bbox"]
        center_x = (x1 + x2) // 2
        head_y = y1 + (y2 - y1) // 4

        wears_helmet = any(hx1 < center_x < hx2 and hy1 < head_y < hy2 for hx1, hy1, hx2, hy2 in helmets)

        if wears_helmet:
            no_helmet_timers[pid]["start"] = None
            no_helmet_timers[pid]["captured"] = False
        else:
            timer = no_helmet_timers[pid]
            if timer["start"] is None:
                timer["start"] = current_time
            elapsed = current_time - timer["start"]

            if not timer["captured"]:
                if elapsed < 5:
                    countdown = int(5 - elapsed)
                    cv2.putText(annotated_frame, f"ID {pid}: ALERT! {countdown}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    filename = f"violation_{uuid.uuid4().hex[:8]}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[ALERT] Pelanggaran: orang ID {pid} tidak pakai helm > 5 detik! Foto disimpan: {filename}")
                    timer["captured"] = True
                    cv2.putText(annotated_frame, f"ID {pid}: PELANGGARAN!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
