import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

model_path = './gesture_recognizer.task'

# Variables globales para almacenar el resultado del gesto
detected_gesture = "No gesture detected"

def print_result(result, output_image, timestamp_ms):
    global detected_gesture  # Permite modificar la variable global
    if result and result.gestures and len(result.gestures[0]) > 0:
        detected_gesture = result.gestures[0][0].category_name
    else:
        detected_gesture = "No gesture detected"

options = vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear imagen para MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Enviar la imagen al reconocedor de gestos
        recognizer.recognize_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        # 📌 Dibujar el gesto detectado en la pantalla
        cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Mostrar el video en tiempo real
        cv2.imshow('Gesture Recognition', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
