from Luz import *
from abc import ABC, abstractmethod


class Landmarks:
    """Clase encargada de estructurar y contener las coordenadas de la mano."""
    
    def __init__(self, lista_coordenadas: list):
        self.__puntos = lista_coordenadas 

    @property
    def puntos(self) -> list:
        return self.__puntos

    def a_diccionario(self) -> list:
        """Facilita el trabajo al módulo de Persistencia (Maximiliano)"""
        return self.__puntos

    @classmethod
    def desde_diccionario(cls, datos: list):
        """Reconstruye el objeto Landmarks desde los datos recuperados de la BD."""
        return cls(datos)

    @classmethod
    def desde_mediapipe(cls, hand_landmarks):
        """
        Método utilitario para que el Módulo de Reconocimiento convierta 
        directamente los objetos nativos de MediaPipe a tu clase.
        """
        lista = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
        return cls(lista)

    def obtener_coordenada(self, indice: int) -> dict:
        """Devuelve las coordenadas (x, y, z) de un landmark específico (0-20)."""
        if 0 <= indice < len(self.__puntos):
            return self.__puntos[indice]
        raise IndexError("Índice de landmark fuera de rango (0-20).")

    def normalizar_respecto_a_muneca(self):
        """
        Resta las coordenadas de la muñeca (índice 0) a todos los puntos.
        Hace que la seña sea independiente de la posición de la mano en la pantalla.
        """
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
        """Imprime o devuelve una descripción de la seña."""
        pass

    @abstractmethod
    def a_diccionario(self) -> dict:
        """Exporta los datos de la seña a un diccionario serializable."""
        pass

    def __eq__(self, otro) -> bool:
        """Permite comparar si dos señas representan lo mismo."""
        if isinstance(otro, Sena):
            return self.__nombre == otro.nombre and self.__idioma == otro.idioma
        return False


class SenaEstatica(Sena):
    """Una seña quieta, representada por una única captura de posición (un solo frame)."""
    
    def __init__(self, nombre: str, idioma: str, posicion: Landmarks):
        super().__init__(nombre, idioma)
        if not isinstance(posicion, Landmarks):
            raise TypeError("El parámetro 'posicion' debe ser una instancia de Landmarks.")
        self.__posicion = posicion  

    @property
    def posicion(self) -> Landmarks:
        return self.__posicion

    def describir(self):
        print(f"SEÑA ESTÁTICA: {self.nombre} | IDIOMA: {self.idioma} | Puntos capturados: {len(self.__posicion.puntos)}")

    def a_diccionario(self) -> dict:
        """Exporta la seña estática a un formato almacenable para la BD."""
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
        """Agrega un frame de movimiento enviado por el Módulo de Reconocimiento."""
        if not isinstance(nuevos_landmarks, Landmarks):
            raise TypeError("Solo se pueden agregar objetos de tipo Landmarks.")
            
        if len(self.__movimiento) < self.__max_frames:
            self.__movimiento.append(nuevos_landmarks)
        else:
            self.__movimiento.pop(0)
            self.__movimiento.append(nuevos_landmarks)

    def limpiar_movimiento(self):
        """Limpia el historial de frames para iniciar una nueva captura."""
        self.__movimiento.clear()

    def describir(self):
        print(f"SEÑA DINÁMICA: {self.nombre} | IDIOMA: {self.idioma} | Frames acumulados: {len(self.__movimiento)}/{self.__max_frames}")

    def a_diccionario(self) -> dict:
        """Exporta la seña dinámica a un formato almacenable para la BD."""
        return {
            "tipo": "dinamica",
            "nombre": self.nombre,
            "idioma": self.idioma,
            "secuencia": [lm.a_diccionario() for lm in self.__movimiento]
        }


class Gesto(Sena):
    """Acción o expresión reconocible de la mano vinculada al sistema (ej: scroll, click)."""
    
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
        """Exporta el gesto mapeado para la base de datos."""
        return {
            "tipo": "gesto",
            "nombre": self.nombre,
            "idioma": self.idioma,
            "tipo_gesto": self.__tipodegesto,
            "seña_asociada": self. __sena_asociada.a_diccionario()
        }