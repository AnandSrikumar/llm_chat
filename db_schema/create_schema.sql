-- ============================================================
-- LLM / RAG POC Database Schema
-- PostgreSQL + pgvector
-- ============================================================

-- ------------------------------------------------------------
-- Extensions
-- ------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- USERS AND CONVO

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



-- FILES

create table if not exists file_storage_types(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT not null UNIQUE
);

CREATE TABLE IF NOT EXISTS files (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    conversation_id BIGINT NOT NULL,

    filename_original TEXT NOT NULL,
    filename_generated TEXT NOT NULL,

    file_content_hash TEXT NOT NULL,

    mime_type TEXT NOT NULL,
    size BIGINT NOT NULL CHECK (size >= 0),

    file_path TEXT NOT NULL,

    file_storage_type BIGINT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_files_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_file_storage
        FOREIGN KEY (file_storage_type)
        REFERENCES file_storage_types(id)
);

CREATE INDEX IF NOT EXISTS idx_files_conversation_id
    ON files(conversation_id);

INSERT INTO file_storage_types (name)
VALUES
    ('local'),
    ('s3'),
    ('azure')
ON CONFLICT (name) DO NOTHING;

GRANT SELECT
ON file_storage_types
TO app_user;

REVOKE UPDATE, DELETE, INSERT
ON file_storage_types
FROM app_user;

-- chunks

CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_id BIGINT NOT NULL,
    chunk_text TEXT NOT NULL,
    cleaned_chunk_text TEXT NOT NULL,

    embedding VECTOR(384),

    CONSTRAINT fk_chunks_file
        FOREIGN KEY (file_id)
        REFERENCES files(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_chunks_file_index
        UNIQUE (file_id, chunk_index)
);