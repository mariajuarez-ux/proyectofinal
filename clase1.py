import cv2
import mediapipe as mp

from Alan import SenaEstatica, SenaDinamica, Gesto


# Ruta del modelo
modelo = "hand_landmarker.task"

# Configurar MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=modelo
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# Abrir la cámara
camara = cv2.VideoCapture(0)

tiempo = 0

with HandLandmarker.create_from_options(opciones) as detector:

    while camara.isOpened():

        correcto, frame = camara.read()

        if not correcto:
            break

        # Convertir la imagen de OpenCV a RGB
        imagen_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imagen_rgb
        )

        # Detectar la mano
        resultado = detector.detect_for_video(
            imagen_mp, tiempo
        )

        tiempo += 33

        # Si encontró una mano
        if resultado.hand_landmarks:

            for mano in resultado.hand_landmarks:

                # Dibujar los 21 puntos
                for punto in mano:

                    x = int(punto.x * frame.shape[1])
                    y = int(punto.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Mano detectada",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No se detecta ninguna mano",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("SEÑA - Camara", frame)

        # Presionar Q para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camara.release()
cv2.destroyAllWindows()
