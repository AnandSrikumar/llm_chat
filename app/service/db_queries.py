FILE_OWNER_QUERY = """
select u.username as username,
c.id AS convo_dir from users u join conversations c on c.owner_id=u.id
where c.id=$1
"""

FILE_STORAGE_ID_QUERY = """
select id from file_storage_types where name=$1
"""
FILE_INSERT_QUERY = """
insert into files (conversation_id, 
        filename_original, 
        filename_generated,
        file_content_hash,
        mime_type,
        size,
        file_path,
        file_storage_type) 
    values 

    ($1, $2, $3, $4, $5, $6, $7, $8)
    returning id

"""

CHUNK_INSERT_QUERY = """
INSERT INTO chunks (
    file_id,
    chunk_index,
    chunk_text,
    cleaned_chunk_text,
    embedding
)
VALUES ($1, $2, $3, $4, $5)

"""
