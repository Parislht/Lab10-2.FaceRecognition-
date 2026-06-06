import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rtree import index
import face_recognition
import time
import psycopg2



DATASET_PATH = r"C:\Users\Paris Herrera\Desktop\utec\2026 - 1\BD2\Semana11\archive\lfw-funneled\lfw_funneled"

QUERY_IMAGE_PATH = r"C:\Users\Paris Herrera\Desktop\utec\2026 - 1\BD2\Semana11\consulta.jpg"


coleccion = []

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "123456"
}


for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True):
    person = os.path.basename(os.path.dirname(path))
    coleccion.append({
        "person": person,
        "path": path
    })

coleccion = pd.DataFrame(coleccion)

print("Total de imágenes encontradas:", len(coleccion))
print(coleccion.head(10))

def mostrarFotos(coleccion, posiciones):
    plt.figure(figsize=(16, 10))

    for i, idx in enumerate(posiciones):
        img = plt.imread(coleccion["path"].iloc[idx])
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(coleccion["person"].iloc[idx] + " " + str(img.shape))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

posiciones = list(range(0, 16))
mostrarFotos(coleccion, posiciones)


def generate_face_embeddings(coleccion, N):

    resultados = []
    errores = 0
    sin_rostro = 0

    inicio = time.time()

    # Nos aseguramos de no pedir más imágenes de las disponibles
    N = min(N, len(coleccion))

    for i in range(N):
        person = coleccion["person"].iloc[i]
        path = coleccion["path"].iloc[i]

        try:
            #Leer imagen
            image = face_recognition.load_image_file(path)

            #Extraer encoding facial
            face_encodings = face_recognition.face_encodings(image)

            #Si detectó al menos un rostro -> tomamos el primero
            if face_encodings:
                embedding = face_encodings[0]

                resultados.append({
                    "person": person,
                    "path": path,
                    "embedding": embedding
                })
            else:
                sin_rostro += 1

        except Exception as e:
            errores += 1
            print(f"Error procesando {path}: {e}")

        #Mensaje de avance cada 100 imágenes
        if (i + 1) % 1000 == 0:
            print(f"Procesadas {i + 1}/{N} imagenes")

    fin = time.time()

    embeddings_df = pd.DataFrame(resultados)

    print("\nResumen:")
    print(f"Imágenes solicitadas: {N}")
    print(f"Embeddings generados: {len(embeddings_df)}")
    print(f"Imágenes sin rostro detectado: {sin_rostro}")
    print(f"Errores: {errores}")
    print(f"Tiempo total: {fin - inicio:.2f} segundos")

    return embeddings_df



#Convierte un numpy array de 128 dimensiones al formato textual que acepta pgvector
def embedding_to_pgvector(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def save_embeddings_to_postgres(embeddings_df, clean_table=False):

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Asegurar que la extensión y tabla existen
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            name TEXT,
            path TEXT,
            embedding VECTOR(128)
        );
    """)

    if clean_table:
        cur.execute("TRUNCATE TABLE face_embeddings RESTART IDENTITY;")

    inserted = 0

    for _, row in embeddings_df.iterrows():
        cur.execute(
            """
            INSERT INTO face_embeddings (name, path, embedding)
            VALUES (%s, %s, %s::vector);
            """,
            (
                row["person"],
                row["path"],
                embedding_to_pgvector(row["embedding"])
            )
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Se insertaron {inserted} embeddings en Postgre")

def save_query_embedding(name, image_path, embedding):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_embeddings (
            id SERIAL PRIMARY KEY,
            name TEXT,
            path TEXT,
            embedding VECTOR(128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute(
        """
        INSERT INTO query_embeddings (name, path, embedding)
        VALUES (%s, %s, %s::vector)
        RETURNING id;
        """,
        (
            name,
            image_path,
            embedding_to_pgvector(embedding)
        )
    )

    query_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    print(f"Consulta guardada con id = {query_id}")
    return query_id

def generate_query_embedding(image_path):
    image = face_recognition.load_image_file(image_path)

    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        raise ValueError("No se detecto ningun rostro en la imagen de consulta")

    if len(face_encodings) > 1:
        print("Advertencia: se detecto mas de un rostro. Se usara el primer rostro encontrado.")

    return face_encodings[0]

N = 14000

embeddings_df = generate_face_embeddings(coleccion, N)#
print(embeddings_df.head())
print(type(embeddings_df["embedding"].iloc[0]))
print(embeddings_df["embedding"].iloc[0].shape)
save_embeddings_to_postgres(embeddings_df, clean_table=True)

query_embedding = generate_query_embedding(QUERY_IMAGE_PATH)

print(type(query_embedding))
print(query_embedding.shape)

query_id = save_query_embedding(
    name="consulta_paris",
    image_path=QUERY_IMAGE_PATH,
    embedding=query_embedding
)

# Datos obtenidos del experimento
sizes = [1000, 2000, 4000, 8000, 12000]

linear_times = [14.270, 7.281, 8.714, 8.594, 11.812]
ivf_times = [68.319, 73.437, 70.201, 68.100, 70.989]
hnsw_times = [122.395, 109.372, 64.385, 79.553, 68.779]

plt.figure(figsize=(10, 6))

plt.plot(sizes, linear_times, marker='o', label='KNN Lineal')
plt.plot(sizes, ivf_times, marker='o', label='KNN con IVF')
plt.plot(sizes, hnsw_times, marker='o', label='KNN con HNSW')

plt.xlabel('Tamaño del dataset / número de embeddings')
plt.ylabel('Tiempo de ejecución (ms)')
plt.title('Comparación de tiempos: KNN Lineal vs IVF vs HNSW')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


