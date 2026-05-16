import cv2
import model as md
import torch
import os
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join('models', 'cnn_asl.pth')

recognizer = md.CNN(device=device)
recognizer.load_state_dict(torch.load(model_path, map_location=device))
recognizer.eval()

box_x, box_y = 150, 100
box_w, box_h = 300, 300

train_root = os.path.join('..', 'data', 'asl_alphabet_train', 'asl_alphabet_train')
CLASSES = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])

print("Покажите жест в квадрате. Чтобы выйти нажмите 'q'.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр.")
        break

    image = frame[box_y:(box_y+box_h)][box_x:(box_x+box_w)]
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.astype(np.float32)/255
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = recognizer(image)
        pred = torch.argmax(output, dim=1).item()
        pred_letter = CLASSES[pred]

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
    text = f"Prediction: {pred_letter}"
    cv2.putText(frame, text, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('ASL Alphabet Recognizer', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()