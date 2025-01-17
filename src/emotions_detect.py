import cv2
from deepface import DeepFace
import os
import csv
from datetime import datetime

def emotion_rec():
    cap = cv2.VideoCapture(0)

    with open("emotions.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Emotion", "Confidence"])

    while cap.isOpened():
        r, frame = cap.read()
        print("Hello")
        if r:
            print("FRAME CAPTURED SUCCESSFULLY")
            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

                print("Analysis Result:", result)

                if isinstance(result, list):
                    result = result[0]
                    
                if "emotion" in result:
                    emotion = result["dominant_emotion"]
                    confidence = result["emotion"][emotion]

                    if confidence > 70:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        with open("emotions.csv", mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, emotion, confidence])

                        cv2.putText(frame, f"Emotion: {emotion} ({confidence:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        if "region" in result:
                            x, y, w, h = result["region"]["x"], result["region"]["y"], result["region"]["w"], result["region"]["h"]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        print(f"Confidence is below 70%. Emotion: {emotion}, Confidence: {confidence:.2f}%")

                else:
                    print("No emotion data found in the result.")

            except Exception as e:
                print(f"An exception has occurred: {e}")
            
            cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_rec()
