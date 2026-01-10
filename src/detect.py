from ultralytics import YOLO
import cv2
import os

def run_detection():
    image_dir = "data/sample_images"
    output_dir = "runs/uav_detect_yolo_labels"

    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model (drone-trained weights)
    model = YOLO("model/dron.pt")

    images = [
        img for img in os.listdir(image_dir)
        if img.lower().endswith((".jpg", ".jpeg", ".png",".webp"))
    ]

    if not images:
        print("❌ No images found")
        return

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)

        results = model(img_path, conf=0.3)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # YOLO-predicted class
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                label = f"{class_name} {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                # Label background
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - h - 6),
                    (x1 + w + 4, y1),
                    (0, 0, 255),
                    -1
                )

                # Put label text
                cv2.putText(
                    image,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)
        print(f"✅ Processed: {img_name}")

    print("\n📁 Results saved in:", output_dir)

if __name__ == "__main__":
    run_detection()

