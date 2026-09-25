import sqlite3
import os

class Usuario:
    """Clase base para todos los usuarios del sistema SENA."""

    AVATARES_DISPONIBLES = ["predeterminado", "oyente", "usuario_señas", "avatar_oyente", "avatar_señas"]

    def __init__(self, nombre: str, velocidad_voz: float = 1.0, avatar: str = "predeterminado"):
        if not nombre or not isinstance(nombre, str):
            raise ValueError("El nombre del usuario no puede estar vacío.")
        
        self.__nombre = nombre.strip()
        self.__velocidad_voz = 1.0
        self.__avatar = "predeterminado"

        self.cambiar_velocidad_voz(velocidad_voz)
        self.cambiar_avatar(avatar)

    def obtener_nombre(self) -> str:
        return self.__nombre

    def obtener_velocidad_voz(self) -> float:
        return self.__velocidad_voz

    def obtener_avatar(self) -> str:
        return self.__avatar

    def cambiar_nombre(self, nuevo_nombre: str) -> bool:
        if nuevo_nombre and isinstance(nuevo_nombre, str):
            self.__nombre = nuevo_nombre.strip()
            return True
        print("Error: Nombre inválido.")
        return False

    def cambiar_velocidad_voz(self, velocidad: float) -> bool:
        if isinstance(velocidad, (int, float)) and 0.5 <= velocidad <= 2.0:
            self.__velocidad_voz = round(float(velocidad), 2)
            return True
        else:
            print("Error: La velocidad de voz debe estar entre 0.5 y 2.0")
            return False

    def cambiar_avatar(self, avatar: str) -> bool:
        if avatar in self.AVATARES_DISPONIBLES:
            self.__avatar = avatar
            return True
        else:
            print(f"Error: Avatar '{avatar}' no válido. Opciones: {self.AVATARES_DISPONIBLES}")
            return False

    def mostrar_perfil(self) -> str:
        return (
            f"Usuario: {self.__nombre}\n"
            f"Tipo: {self.__class__.__name__}\n"
            f"Velocidad de voz: {self.__velocidad_voz}x\n"
            f"Avatar: {self.__avatar}"
        )

    def a_diccionario(self) -> dict:
        return {
            "nombre": self.__nombre,
            "tipo": self.__class__.__name__,
            "velocidad_voz": self.__velocidad_voz,
            "avatar": self.__avatar
        }

    @staticmethod
    def _inicializar_tabla(ruta_db: str = "sena.db"):
        try:
            with sqlite3.connect(ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS usuarios (
                        nombre TEXT PRIMARY KEY,
                        velocidad_voz REAL NOT NULL,
                        avatar TEXT NOT NULL,
                        tipo TEXT NOT NULL
                    )
                ''')
                conexion.commit()
        except sqlite3.Error as e:
            print(f"Error al conectar con la Base de Datos: {e}")

    def guardar_en_db(self, ruta_db: str = "sena.db") -> bool:
        self._inicializar_tabla(ruta_db)
        tipo_usuario = self.__class__.__name__
        try:
            with sqlite3.connect(ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute('''
                    INSERT INTO usuarios (nombre, velocidad_voz, avatar, tipo)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(nombre) DO UPDATE SET
                        velocidad_voz=excluded.velocidad_voz,
                        avatar=excluded.avatar,
                        tipo=excluded.tipo
                ''', (self.__nombre, self.__velocidad_voz, self.__avatar, tipo_usuario))
                conexion.commit()
                print(f"Perfil de '{self.__nombre}' guardado exitosamente.")
                return True
        except sqlite3.Error as e:
            print(f"Error al guardar usuario en SQLite: {e}")
            return False

    @classmethod
    def cargar_de_db(cls, nombre: str, ruta_db: str = "sena.db"):
        cls._inicializar_tabla(ruta_db)
        try:
            with sqlite3.connect(ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute("SELECT nombre, velocidad_voz, avatar, tipo FROM usuarios WHERE nombre = ?", (nombre,))
                fila = cursor.fetchone()
                
                if fila:
                    nombre_db, velocidad, avatar, tipo = fila
                    if tipo == "UsuarioOyente":
                        return UsuarioOyente(nombre_db, velocidad, avatar)
                    elif tipo == "UsuarioSenas":
                        return UsuarioSenas(nombre_db, velocidad, avatar)
                    else:
                        return Usuario(nombre_db, velocidad, avatar)
                else:
                    print(f"No se encontró el usuario '{nombre}'.")
                    return None
        except sqlite3.Error as e:
            print(f"Error al cargar usuario de SQLite: {e}")
            return None

    @staticmethod
    def listar_todos_los_usuarios(ruta_db: str = "sena.db") -> list:
        Usuario._inicializar_tabla(ruta_db)
        try:
            with sqlite3.connect(ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute("SELECT nombre FROM usuarios")
                return [fila[0] for fila in cursor.fetchall()]
        except sqlite3.Error:
            return []

    @staticmethod
    def eliminar_de_db(nombre: str, ruta_db: str = "sena.db") -> bool:
        Usuario._inicializar_tabla(ruta_db)
        try:
            with sqlite3.connect(ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute("DELETE FROM usuarios WHERE nombre = ?", (nombre,))
                conexion.commit()
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False


class UsuarioOyente(Usuario):
    """Usuario que principalmente recibe información mediante voz."""

    def __init__(self, nombre: str, velocidad_voz: float = 1.0, avatar: str = "avatar_oyente"):
        super().__init__(nombre, velocidad_voz, avatar)

    def mostrar_perfil(self) -> str:
        return (
            "=== PERFIL OYENTE ===\n"
            f"Nombre: {self.obtener_nombre()}\n"
            f"Velocidad de voz: {self.obtener_velocidad_voz()}x\n"
            f"Avatar: {self.obtener_avatar()}"
        )


class UsuarioSenas(Usuario):
    """Usuario que utiliza principalmente la lengua de señas."""

    def __init__(self, nombre: str, velocidad_voz: float = 1.0, avatar: str = "avatar_señas"):
        super().__init__(nombre, velocidad_voz, avatar)

    def mostrar_perfil(self) -> str:
        return (
            "=== PERFIL USUARIO DE SEÑAS ===\n"
            f"Nombre: {self.obtener_nombre()}\n"
            f"Velocidad de voz: {self.obtener_velocidad_voz()}x\n"
            f"Avatar: {self.obtener_avatar()}"
        )

if __name__ == "__main__":
    print("--- 1. Creando Usuarios ---")
    u1 = UsuarioOyente("Maxi", velocidad_voz=1.2)
    u2 = UsuarioSenas("Sergio", velocidad_voz=0.8)

    print(u1.mostrar_perfil())
    print()
    print(u2.mostrar_perfil())
    print()

    print("--- 2. Guardando en SQLite ---")
    u1.guardar_en_db()
    u2.guardar_en_db()

    print("\n--- 3. Listar todos los usuarios en la DB ---")
    print("Usuarios registrados:", Usuario.listar_todos_los_usuarios())

    print("\n--- 4. Recuperar un usuario de SQLite ---")
    u_cargado = Usuario.cargar_de_db("Sergio")
    if u_cargado:
        print(u_cargado.mostrar_perfil())