import paho.mqtt.client as mqtt
import sqlite3
from datetime import datetime

# Función para manejar el evento de conexión
def on_connect(client, userdata, flags, rc):
    print("Conectado al broker MQTT con resultado: " + mqtt.connack_string(rc))
    # Suscribirse a un tema
    client.subscribe("/alerta")

# Función para manejar el evento de recepción de mensaje
def on_message(client, userdata, msg):
    print("Mensaje recibido: " + msg.payload.decode())
    guardar_mensaje(msg.payload.decode())

# Función para guardar el mensaje en la base de datos
def guardar_mensaje(mensaje):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO mensajes (timestamp, mensaje) VALUES (?, ?)", (timestamp, mensaje))
    conexion.commit()

# Conexión a la base de datos y creación de la tabla
conexion = sqlite3.connect('mensajes.db')
cursor = conexion.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS mensajes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    mensaje TEXT
                  )''')
conexion.commit()

# Configurar el cliente MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Conectar al broker MQTT
client.connect("localhost", 1883, 60)

# Iniciar el bucle de red del cliente MQTT
client.loop_forever()
