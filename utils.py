from ultralytics import YOLO
import cv2

def load_model(pretrained_weights_path: str):
    return YOLO(pretrained_weights_path)

def predict_and_detect(pretrained_model, file: str):
    image = file.copy()
    prediction = pretrained_model(image)
    for result in prediction:
        for box in result.boxes:
            print(box)
            xmin_ymin = (int(box.xyxy[0][0]), 
                        int(box.xyxy[0][1]))
             
            xmax_ymax = (int(box.xyxy[0][2]), 
                        int(box.xyxy[0][3]))
            
            cv2.rectangle(image, xmin_ymin, xmax_ymax, (255, 0, 0), 2)
            cv2.putText(image, 
                        f"{result.names[int(box.cls[0])]}, {box.conf.item():.2f}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1)
    return image, prediction
    
