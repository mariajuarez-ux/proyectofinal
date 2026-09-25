from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

class ReconocedordeSeñas(ABC):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands( static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
    
    @abstractmethod
    def reconocer(self, frame_capturado):
        pass
    
    def procesar_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(frame_rgb)
        return resultados

class ReconocidorEstatico(ReconocedordeSeñas):
    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Sena estatica detectada"

class ReconocidorDinamico(ReconocedordeSeñas):
    def __init__(self):
        super().__init__()
        self.historial_frames = [] 

    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Sena dinamica detectada"

class ReconocidorAlfabeto(ReconocedordeSeñas):
    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Letra detectada"

import cv2
import mediapipe as mp
from abc import ABC, abstractmethod


class ReconocedorDeSeñas(ABC):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    @abstractmethod
    def reconocer(self, frame):
        pass


class ReconocidorEstatico(ReconocedorDeSeñas):
    def reconocer(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.hands.process(img_rgb)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
            return "Mano detectada"

        return "Buscando mano..."


cap = cv2.VideoCapture(0) 
reconocedor = ReconocidorEstatico()

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    estado = reconocedor.reconocer(frame)

    cv2.putText(
        frame,
        estado,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Prueba Modulo Reconocimiento", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
