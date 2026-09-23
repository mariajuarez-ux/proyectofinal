from abc import ABC, abstractmethod
import math
import cv2
import mediapipe as mp


class Landmarks:
    """Clase encargada de estructurar y contener las coordenadas de la mano."""
    
    def __init__(self, lista_coordenadas: list):
        self.__puntos = lista_coordenadas 

    @property
    def puntos(self) -> list:
        return self.__puntos

    def a_diccionario(self) -> list:
        return self.__puntos

    @classmethod
    def desde_diccionario(cls, datos: list):
        return cls(datos)

    @classmethod
    def desde_mediapipe(cls, hand_landmarks):
        lista = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
        return cls(lista)

    def obtener_coordenada(self, indice: int) -> dict:
        if 0 <= indice < len(self.__puntos):
            return self.__puntos[indice]
        raise IndexError("Índice de landmark fuera de rango (0-20).")

    def normalizar_respecto_a_muneca(self):
        if not self.__puntos:
            return
        muneca = self.__puntos[0]
        self.__puntos = [
            {
                'x': p['x'] - muneca['x'],
                'y': p['y'] - muneca['y'],
                'z': p['z'] - muneca['z']
            } for p in self.__puntos
        ]


class Sena(ABC):
    """Clase base abstracta para todas las señas del sistema."""
    
    def __init__(self, nombre: str, idioma: str):
        self.__nombre = nombre
        self.__idioma = idioma

    @property
    def nombre(self) -> str:
        return self.__nombre

    @property
    def idioma(self) -> str:
        return self.__idioma

    @abstractmethod
    def describir(self):
        pass

    @abstractmethod
    def a_diccionario(self) -> dict:
        pass

    def __eq__(self, otro) -> bool:
        if isinstance(otro, Sena):
            return self.__nombre == otro.nombre and self.__idioma == otro.idioma
        return False


class SenaEstatica(Sena):
    """Una seña quieta, representada por una única captura de posición."""
    
    def __init__(self, nombre: str, idioma: str, posicion: Landmarks):
        super().__init__(nombre, idioma)
        if not isinstance(posicion, Landmarks):
            raise TypeError("El parámetro 'posicion' debe ser una instancia de Landmarks.")
        self.__posicion = posicion  

    @property
    def posicion(self) -> Landmarks:
        return self.__posicion

    def describir(self):
        print(f"SEÑA ESTÁTICA: {self.nombre} | IDIOMA: {self.idioma} | Puntos: {len(self.__posicion.puntos)}")

    def a_diccionario(self) -> dict:
        return {
            "tipo": "estatica",
            "nombre": self.nombre,
            "idioma": self.idioma,
            "posicion": self.__posicion.a_diccionario()
        }


class SenaDinamica(Sena):
    """Una seña que se mueve, representada por una secuencia temporal de Landmarks."""
    
    def __init__(self, nombre: str, idioma: str, max_frames: int = 30):
        super().__init__(nombre, idioma)
        self.__max_frames = max_frames
        self.__movimiento: list[Landmarks] = []  

    @property
    def movimiento(self) -> list[Landmarks]:
        return self.__movimiento

    def agregar_frame(self, nuevos_landmarks: Landmarks):
        if not isinstance(nuevos_landmarks, Landmarks):
            raise TypeError("Solo se pueden agregar objetos de tipo Landmarks.")
            
        if len(self.__movimiento) < self.__max_frames:
            self.__movimiento.append(nuevos_landmarks)
        else:
            self.__movimiento.pop(0)
            self.__movimiento.append(nuevos_landmarks)

    def limpiar_movimiento(self):
        self.__movimiento.clear()

    def describir(self):
        print(f"SEÑA DINÁMICA: {self.nombre} | IDIOMA: {self.idioma} | Frames: {len(self.__movimiento)}/{self.__max_frames}")

    def a_diccionario(self) -> dict:
        return {
            "tipo": "dinamica",
            "nombre": self.nombre,
            "idioma": self.idioma,
            "secuencia": [lm.a_diccionario() for lm in self.__movimiento]
        }


class Gesto(Sena):
    """Acción o expresión reconocible de la mano vinculada al sistema."""
    
    def __init__(self, nombre: str, idioma: str, tipodegesto: str, sena_asociada: Sena):
        super().__init__(nombre, idioma)
        if not isinstance(sena_asociada, Sena):
            raise TypeError("El parámetro 'sena_asociada' debe ser una instancia de Sena.")
        self.__tipodegesto = tipodegesto
        self.__sena_asociada = sena_asociada  

    @property
    def tipodegesto(self) -> str:
        return self.__tipodegesto

    @property
    def sena_asociada(self) -> Sena:
        return self.__sena_asociada

    def describir(self):
        print(f"GESTO DE INTERFAZ: {self.nombre} | Tipo: {self.__tipodegesto} | Disparado por seña: {self.__sena_asociada.nombre}")

    def a_diccionario(self) -> dict:
        return {
            "tipo": "gesto",
            "nombre": self.nombre,
            "idioma": self.idioma,
            "tipo_gesto": self.__tipodegesto,
            "seña_asociada": self.__sena_asociada.a_diccionario()
        }


class ReconocedorDeSeñas(ABC):
    """Clase base abstracta para procesar video y capturar landmarks."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_dibujo = self.mp_hands.HAND_CONNECTIONS

    @abstractmethod
    def reconocer(self, frame):
        pass

    def procesar_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def dibujar_landmarks(self, frame, resultados):
        if resultados.multi_hand_landmarks:
            for mano in resultados.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, mano, self.mp_dibujo)
        return frame


class ReconocedorEstatico(ReconocedorDeSeñas):
    """Reconocedor que evalúa la postura de la mano para identificar gestos cotidianos, números y abecedario."""

    def _distancia(self, p1: dict, p2: dict) -> float:
        """Calcula la distancia 2D entre dos puntos de la mano."""
        return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)

    def _obtener_dedos_levantados(self, puntos: list) -> list:
        """Determina qué dedos están abiertos (1) o cerrados (0)."""
        dedos = []

        # Pulgar (evaluado horizontalmente)
        if abs(puntos[4]['x'] - puntos[2]['x']) > 0.04:
            dedos.append(1)
        else:
            dedos.append(0)

        # Índice, Medio, Anular, Meñique (evaluados verticalmente)
        puntas = [8, 12, 16, 20]
        for tip in puntas:
            if puntos[tip]['y'] < puntos[tip - 2]['y']:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

    def clasificar_gesto(self, dedos: list, puntos: list) -> str:
        """Identifica expresiones cotidianas, números y el abecedario completo A-Z."""
        
        # Distancias calculadas entre articulaciones clave
        d_pulgar_indice = self._distancia(puntos[4], puntos[8])
        d_pulgar_medio = self._distancia(puntos[4], puntos[12])
        d_pulgar_anular = self._distancia(puntos[4], puntos[16])
        d_pulgar_menique = self._distancia(puntos[4], puntos[20])
        d_indice_medio = self._distancia(puntos[8], puntos[12])

        # ==========================================
        # 1. GESTOS Y EXPRESIONES COTIDIANAS
        # ==========================================

        if dedos == [1, 0, 0, 0, 0] and puntos[4]['y'] < puntos[3]['y'] and puntos[4]['y'] < puntos[6]['y']:
            return "Pulgar Arriba / BIEN / SÍ"

        if dedos == [1, 0, 0, 0, 0] and puntos[4]['y'] > puntos[3]['y'] and puntos[4]['y'] > puntos[6]['y']:
            return "Pulgar Abajo / MAL"

        if d_pulgar_indice < 0.05 and d_pulgar_medio < 0.05 and dedos[3:] == [0, 0]:
            return "Gesto: NO"

        if dedos == [0, 1, 1, 1, 1] and puntos[8]['y'] < puntos[5]['y']:
            return "Gesto: Gracias"

        if dedos == [0, 0, 0, 0, 0] and puntos[4]['y'] < puntos[10]['y'] and puntos[4]['x'] > puntos[6]['x']:
            return "Gesto: Perdón"

        if dedos == [1, 0, 0, 0, 1]:
            return "Gesto: Chau / Saludo"

        # ==========================================
        # 2. NÚMEROS (0 AL 10)
        # ==========================================

        # Número 0: Círculo cerrado con la mano
        if d_pulgar_indice < 0.04 and d_pulgar_medio < 0.05 and dedos == [0, 0, 0, 0, 0]:
            return "Número 0"

        # Número 1: Solo el índice levantado (sin extensión lateral de pulgar)
        if dedos == [0, 1, 0, 0, 0] and d_pulgar_medio > 0.08:
            return "Número 1"

        # Número 2: Índice y Medio levantados
        if dedos == [0, 1, 1, 0, 0] and d_indice_medio >= 0.04:
            return "Número 2"

        # Número 3: Pulgar, Índice y Medio extendidos
        if dedos == [1, 1, 1, 0, 0] and d_indice_medio >= 0.03:
            return "Número 3"

        # Número 4: Cuatro dedos estirados sin el pulgar
        if dedos == [0, 1, 1, 1, 1]:
            return "Número 4"

        # Número 5 / Hola: Todos los dedos abiertos
        if dedos == [1, 1, 1, 1, 1] and d_pulgar_indice > 0.12:
            return "Número 5 / Hola"

        # Número 6: Pulgar tocando el Meñique
        if d_pulgar_menique < 0.05 and dedos[1:4] == [1, 1, 1]:
            return "Número 6"

        # Número 7: Pulgar tocando el Anular
        if d_pulgar_anular < 0.05 and dedos[1:3] == [1, 1] and dedos[4] == 1:
            return "Número 7"

        # Número 8: Pulgar tocando el Medio
        if d_pulgar_medio < 0.05 and dedos[1] == 1 and dedos[3:] == [1, 1]:
            return "Número 8"

        # Número 9: Pulgar tocando el Índice
        if d_pulgar_indice < 0.05 and dedos[2:] == [1, 1, 1]:
            return "Número 9"

        # Número 10: Puño agitando el pulgar (evaluación estática)
        if dedos == [1, 0, 0, 0, 0] and puntos[4]['x'] > puntos[3]['x']:
            return "Número 10"

        # ==========================================
        # 3. ABECEDARIO COMPLETO (A - Z)
        # ==========================================

        if dedos == [1, 0, 0, 0, 0] and puntos[4]['y'] < puntos[6]['y']:
            return "Letra A"

        if dedos == [0, 1, 1, 1, 1]:
            return "Letra B"

        if dedos == [1, 1, 1, 1, 1] and 0.07 < d_pulgar_indice < 0.16 and puntos[8]['y'] > puntos[6]['y'] - 0.02:
            return "Letra C"

        if dedos == [0, 1, 0, 0, 0] and d_pulgar_medio < 0.06:
            return "Letra D"

        if dedos == [0, 0, 0, 0, 0] and puntos[8]['y'] > puntos[6]['y']:
            return "Letra E"

        if d_pulgar_indice < 0.05 and dedos[2:] == [1, 1, 1]:
            return "Letra F"

        if dedos == [1, 1, 0, 0, 0] and d_pulgar_indice < 0.07:
            return "Letra G"

        if dedos == [0, 1, 1, 0, 0] and d_indice_medio < 0.04 and puntos[8]['x'] > puntos[6]['x']:
            return "Letra H"

        if dedos == [0, 0, 0, 0, 1]:
            return "Letra I"

        # Letra J: Meñique levantado apuntando al costado
        if dedos == [0, 0, 0, 0, 1] and puntos[20]['x'] < puntos[18]['x']:
            return "Letra J"

        if dedos == [1, 1, 1, 0, 0] and puntos[12]['y'] > puntos[8]['y']:
            return "Letra K"

        if dedos == [1, 1, 0, 0, 0] and d_pulgar_indice > 0.12:
            return "Letra L"

        if dedos == [0, 0, 0, 0, 0] and puntos[4]['x'] < puntos[16]['x']:
            return "Letra M"

        if dedos == [0, 0, 0, 0, 0] and puntos[4]['x'] < puntos[12]['x']:
            return "Letra N"

        if d_pulgar_indice < 0.05 and dedos[2:] == [0, 0, 0]:
            return "Letra O"

        if dedos == [0, 1, 1, 0, 0] and puntos[8]['y'] > puntos[5]['y']:
            return "Letra P"

        if dedos == [1, 1, 0, 0, 0] and puntos[8]['y'] > puntos[5]['y']:
            return "Letra Q"

        if dedos == [0, 1, 1, 0, 0] and puntos[8]['x'] > puntos[12]['x']:
            return "Letra R"

        if dedos == [0, 0, 0, 0, 0] and puntos[4]['y'] < puntos[10]['y']:
            return "Letra S"

        if dedos == [0, 0, 0, 0, 0] and puntos[4]['x'] < puntos[8]['x']:
            return "Letra T"

        if dedos == [0, 1, 1, 0, 0] and d_indice_medio < 0.04:
            return "Letra U"

        if dedos == [0, 1, 1, 0, 0] and d_indice_medio >= 0.04:
            return "Letra V"

        if dedos == [0, 1, 1, 1, 0]:
            return "Letra W"

        if dedos == [0, 1, 0, 0, 0] and puntos[8]['y'] > puntos[7]['y']:
            return "Letra X"

        if dedos == [1, 0, 0, 0, 1]:
            return "Letra Y"

        # Letra Z: Índice extendido hacia el lateral trazando la Z
        if dedos == [0, 1, 0, 0, 0] and puntos[8]['x'] < puntos[6]['x']:
            return "Letra Z"

        return "Buscando Seña / Desconocido"

    def reconocer(self, frame):
        resultados = self.procesar_landmarks(frame)
        frame = self.dibujar_landmarks(frame, resultados)

        if resultados.multi_hand_landmarks:
            for mano in resultados.multi_hand_landmarks:
                mis_landmarks = Landmarks.desde_mediapipe(mano)
                
                dedos = self._obtener_dedos_levantados(mis_landmarks.puntos)
                nombre_sena = self.clasificar_gesto(dedos, mis_landmarks.puntos)

                sena_detectada = SenaEstatica(nombre=nombre_sena, idioma="LSA", posicion=mis_landmarks)

                # Visualización del texto sobre el video
                cv2.rectangle(frame, (15, 15), (480, 65), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"SEÑA: {sena_detectada.nombre}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        else:
            cv2.putText(
                frame,
                "Buscando mano...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        return frame

    
