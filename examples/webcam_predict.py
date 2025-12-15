# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

import argparse

import cv2

from ultralytics import YOLO

# COCO class IDs of interest:
# 0=person, 63=laptop, 67=cell phone, 47=cup, 73=book
TARGET_CLASS_IDS = [0, 63, 67, 47, 73]


def run(
    model_path: str = "yolov8m.pt",
    conf: float = 0.7,
    device: str = "cpu",
    cam_index: int = 0,
    width: int | None = None,
    height: int | None = None,
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera {cam_index}")
        return

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    names = model.names
    print("Pressione 'q' para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, device=device, conf=conf, verbose=False)
        r = results[0]

        for box, cls, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if float(score) < conf:
                continue
            cls_id = int(cls)
            if cls_id not in TARGET_CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{names[cls_id]} {float(score):.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLOv8 - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Detecção em tempo real com webcam usando YOLOv8")
    p.add_argument("--model", default="yolov8n.pt", help="Caminho para pesos YOLOv8 (.pt)")
    p.add_argument("--conf", type=float, default=0.7, help="Threshold de confiança")
    p.add_argument("--device", default="cpu", help="Dispositivo: 'cpu' ou 'cuda'")
    p.add_argument("--cam", type=int, default=0, help="Índice da câmera (0 padrão)")
    p.add_argument("--width", type=int, default=None, help="Largura desejada do frame")
    p.add_argument("--height", type=int, default=None, help="Altura desejada do frame")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.conf, args.device, args.cam, args.width, args.height)
