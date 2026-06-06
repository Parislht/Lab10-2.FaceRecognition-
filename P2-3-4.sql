CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS face_embeddings;

CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    path TEXT,
    embedding VECTOR(128)
);

SELECT COUNT(*) FROM face_embeddings;

SELECT *
FROM face_embeddings
LIMIT 10;


--Generar una tabla para guardar las imagenes de consulta
CREATE TABLE IF NOT EXISTS query_embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    path TEXT,
    embedding VECTOR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT *
FROM query_embeddings
LIMIT 10;


--KNN lineal con distancia euclidiana
EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 10;

--KNN lineal con distancia coseno
EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <=> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <=> q.embedding
LIMIT 10;


--P3

--Limpiar indices previos 
DROP INDEX IF EXISTS idx_face_embedding_l2_ivfflat;
DROP INDEX IF EXISTS idx_face_embedding_cosine_ivfflat;
DROP INDEX IF EXISTS idx_face_embedding_l2_hnsw;
DROP INDEX IF EXISTS idx_face_embedding_cosine_hnsw;




--Indice IVFFlat para distancia euclidiana
CREATE INDEX idx_face_embedding_l2_ivfflat ON face_embeddings 
USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

ANALYZE face_embeddings;

SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 10;


--Indice IVFFlat para distancia euclidiana
CREATE INDEX idx_face_embedding_cosine_ivfflat
ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

ANALYZE face_embeddings;

SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <=> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <=> q.embedding
LIMIT 10;


--P4 


-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 10;


--KNN Indexado - D Euclidiana
DROP INDEX IF EXISTS idx_face_embedding_cosine_ivfflat;
DROP INDEX IF EXISTS idx_face_embedding_l2_hnsw;
DROP INDEX IF EXISTS idx_face_embedding_cosine_hnsw;

CREATE INDEX IF NOT EXISTS idx_face_embedding_l2_hnsw
ON face_embeddings
USING hnsw (embedding vector_l2_ops);

ANALYZE face_embeddings;

-- KNN con IVF usando distancia euclidiana
SET enable_seqscan = off;
SET ivfflat.probes = 10;

SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 10;


--kNN Indexado con HNSW 

DROP INDEX IF EXISTS idx_face_embedding_l2_ivfflat;
DROP INDEX IF EXISTS idx_face_embedding_cosine_ivfflat;
DROP INDEX IF EXISTS idx_face_embedding_cosine_hnsw;

CREATE INDEX IF NOT EXISTS idx_face_embedding_l2_hnsw
ON face_embeddings
USING hnsw (embedding vector_l2_ops);

ANALYZE face_embeddings;


-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 10;




--Comparacion de tiempos Lineal, IvF y HNSW

--Lineal 

--k = 1000

-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 1000;

--k = 2000

-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 2000;

--k = 4000
-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 4000;

--k = 8000

-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 8000;


--k = 12000

-- KNN lineal exacto con distancia euclidiana
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 12000;


--IVF

DROP INDEX IF EXISTS idx_face_embedding_l2_hnsw;

CREATE INDEX IF NOT EXISTS idx_face_embedding_l2_ivfflat
ON face_embeddings
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

--k = 1000
-- KNN con IVFFlat usando distancia euclidiana
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 1000;

--k = 2000
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 2000;

--k = 4000
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 4000;


--k = 8000
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 8000;

--k = 12000
SET enable_seqscan = off;
SET ivfflat.probes = 10;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 12000;


--HNSW
DROP INDEX IF EXISTS idx_face_embedding_l2_ivfflat;

CREATE INDEX IF NOT EXISTS idx_face_embedding_l2_hnsw
ON face_embeddings
USING hnsw (embedding vector_l2_ops);

ANALYZE face_embeddings;
--k = 1000
-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 1000;

--k = 2000
-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 2000;

--k = 4000
-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 4000;

--k = 8000
-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 8000;

--k = 12000
-- KNN con HNSW usando distancia euclidiana
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT 
    f.id,
    f.name,
    f.path,
    f.embedding <-> q.embedding AS distance
FROM face_embeddings f
CROSS JOIN query_embeddings q
WHERE q.id = 1
ORDER BY f.embedding <-> q.embedding
LIMIT 12000;















