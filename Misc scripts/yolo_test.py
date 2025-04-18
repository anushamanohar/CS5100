import os
import sys
import cv2
import numpy as np
import torch
import pybullet as p
from env_setup import make_env


YOLOV5_PATH = "/home/shruti/CS5100/RL_training/fai_data_set/fai_data_set/yolov5"
MODEL_PATH = os.path.join(YOLOV5_PATH, "runs/train/exp/weights/best.pt")

=
sys.path.insert(0, YOLOV5_PATH)


from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.augmentations import letterbox


env = make_env(render=False)()
env = env.unwrapped 
obs, _ = env.reset()
image = obs['image']
img_path = "robot_view.png"
cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f" Saved camera image to {img_path}")


device = select_device('cpu')
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640
conf_thres = 0.4
iou_thres = 0.45

# Preprocess image for YOLOv5 
img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
img_letterboxed = letterbox(img, img_size, stride=stride, auto=True)[0]
img_input = img_letterboxed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img_input = np.ascontiguousarray(img_input)

img_tensor = torch.from_numpy(img_input).to(device)
img_tensor = img_tensor.float() / 255.0
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)


pred = model(img_tensor, augment=False, visualize=False)
pred = non_max_suppression(pred, conf_thres, iou_thres)

# Draw bounding boxes 
detected_objects = []

for det in pred:
    if len(det):
        gain = min(img_size / img.shape[0], img_size / img.shape[1])
        pad_x = (img_size - img.shape[1] * gain) / 2
        pad_y = (img_size - img.shape[0] * gain) / 2

        for i in range(len(det)):
            det[i, [0, 2]] -= pad_x
            det[i, [1, 3]] -= pad_y
            det[i, :4] /= gain
            det[i, :4] = det[i, :4].round()

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = names[int(cls)]

            Z = 0.4
            fx = fy = 84 / (2 * np.tan(np.radians(60 / 2)))
            cx0 = cy0 = 84 // 2
            X = (cx - cx0) * Z / fx
            Y = (cy - cy0) * Z / fy

            detected_objects.append({
                "class": class_name,
                "conf": float(conf),
                "bbox": (x1, y1, x2, y2),
                "center_px": (cx, cy),
                "estimated_3d": (X, Y, Z)
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Display all detections 
print(f"\n Total Detected Objects: {len(detected_objects)}")
for i, obj in enumerate(detected_objects):
    print(f"\n🔹 Object {i+1}:")
    print(f"   Class: {obj['class']}")
    print(f"   Confidence: {obj['conf']:.2f}")
    print(f"   Center (pixels): {obj['center_px']}")
    print(f"   Estimated 3D position: {obj['estimated_3d']}")

# Save annotated image
cv2.imwrite("detected_view.png", img)
print("Detection result saved to detected_view.png")

#  Print robot EE position 
ee_index = env.ee_link_index
ee_pos = p.getLinkState(env.robot_id, ee_index)[0]
print(f"\nRobot End-Effector Position: X={ee_pos[0]:.2f}, Y={ee_pos[1]:.2f}, Z={ee_pos[2]:.2f}")

env.close()
