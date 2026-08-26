-- ============================================================
-- LLM / RAG POC Database Schema
-- PostgreSQL + pgvector
-- ============================================================

-- ------------------------------------------------------------
-- Extensions
-- ------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;


-- ------------------------------------------------------------
-- Users
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id BIGINT NOT NULL,
    convo_name TEXT NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    compaction JSONB NOT NULL DEFAULT '[]'::jsonb,

    CONSTRAINT fk_conversations_owner
        FOREIGN KEY (owner_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    CONSTRAINT chk_messages_array
        CHECK (jsonb_typeof(messages) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner_id
    ON conversations(owner_id);



-- ------------------------------------------------------------
-- Files
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS files (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    mime_type TEXT NOT NULL,
    size BIGINT NOT NULL CHECK (size >= 0),
    file_name_original TEXT NOT NULL,
    file_name TEXT NOT NULL UNIQUE,

    CONSTRAINT fk_files_convo
        FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_files_convo_id
    ON files(conversation_id);


-- ------------------------------------------------------------
-- Chunk configuration
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunk_configs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap INTEGER NOT NULL CHECK (chunk_overlap >= 0),
    chunk_type TEXT NOT NULL,

    CONSTRAINT chk_chunk_overlap
        CHECK (chunk_overlap < chunk_size)
);


-- ------------------------------------------------------------
-- Chunks
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_id BIGINT NOT NULL,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    bm25_chunk_text TEXT,
    chunk_config_id BIGINT,

    CONSTRAINT fk_chunks_file
        FOREIGN KEY (file_id)
        REFERENCES files(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_chunks_config
        FOREIGN KEY (chunk_config_id)
        REFERENCES chunk_configs(id)
        ON DELETE SET NULL,

    CONSTRAINT chk_page_number
        CHECK (page_number IS NULL OR page_number > 0)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id
    ON chunks(file_id);

CREATE INDEX IF NOT EXISTS idx_chunks_file_page
    ON chunks(file_id, page_number);


-- ------------------------------------------------------------
-- Vector embeddings
-- ------------------------------------------------------------
-- Currently assuming 384-dimensional embeddings.
--
-- Example model:
-- all-MiniLM-L6-v2 -> 384 dimensions
--
-- Change vector(384) if you use another model.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id BIGINT PRIMARY KEY,
    embedding VECTOR(384) NOT NULL,

    CONSTRAINT fk_vectors_chunk
        FOREIGN KEY (chunk_id)
        REFERENCES chunks(id)
        ON DELETE CASCADE
);


-- ------------------------------------------------------------
-- Conversations
-- ------------------------------------------------------------


-- ------------------------------------------------------------
-- Vector index
-- ------------------------------------------------------------
-- For the POC, you technically don't need this.
--
-- HNSW gives us approximate nearest-neighbor search and will
-- become useful as the number of embeddings grows.
--
-- cosine distance operator: <=>
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_vectors_embedding_hnsw
    ON vectors
    USING hnsw (embedding vector_cosine_ops);


-- ============================================================
-- Done
-- ============================================================