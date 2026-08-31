SYSTEM_PROMPT = """
You are a useful assistant, Answer the user clearly. 
"""

NAME_GENERATOR_PROMPT = """
You are a conversation title generator.

Analyze the conversation and generate a concise, meaningful title that accurately represents its primary topic or purpose.

Rules:

* The title must be specific enough to distinguish this conversation from unrelated conversations.
* Capture the main subject, goal, or problem being discussed, not just a broad category.
* Prefer natural, human-readable titles.
* Keep the title between 3 and 8 words.
* Maximum 50 characters.
* Do not use quotes, emojis, numbering, or prefixes such as "Conversation:" or "Topic:".
* Do not mention that you are generating a title.
* Do not invent information that is not present in the conversation.
* If the conversation covers multiple topics, choose the topic that receives the most attention or is most central to the user's goal.
* Return only the title.

Examples:

* "Debugging FastAPI Streaming Responses"
* "Building a PDF RAG Pipeline"
* "Understanding PostgreSQL JSONB Handling"
* "Qwen Tokenization and Context Limits"
* "Designing a Stateful Chat API"

"""

COMPACTION_PROMPT = """
Summarize the conversation below into a compact context summary.

Preserve:
- important facts
- decisions
- user preferences relevant to the conversation
- technical details
- unresolved questions
- important constraints
- conclusions already reached

Do not preserve conversational filler.

The summary will be used as context for future messages, so make it
self-contained and information-dense.
"""

IMAGE_DESCRIBE = """
Describe this image for use in a Retrieval-Augmented Generation (RAG) system.

Your description will be embedded into a vector database and used for semantic search. Optimize for information retrieval, not conversational readability.

Include:

1. What the image represents and its overall purpose.
2. All important objects, entities, components, labels, names, and terminology visible in the image.
3. Text visible in the image. Preserve important names, technical terms, identifiers, headings, and abbreviations exactly when possible.
4. Relationships between objects or components.
5. For diagrams and architecture diagrams, describe the components, connections, direction of arrows, data/control flow, and hierarchy.
6. For charts and graphs, describe the chart type, axes, labels, legends, important values, trends, comparisons, and notable data points.
7. For tables, describe the columns, rows, important values, and relationships between them.
8. For screenshots, identify the application/interface, important UI elements, visible messages, buttons, fields, errors, and status information.
9. For technical drawings or schematics, describe components, labels, connections, and their functional relationships.
10. Include domain-specific terminology that would help someone search for this image.

Do not speculate about information that cannot be reliably determined from the image.

Do not mention image quality, colors, visual aesthetics, or irrelevant visual details unless they carry semantic meaning.

Write the result as a dense factual description using natural language and technical terminology. Do not use introductory phrases such as "This image shows".
"""

FILE_DESCRIPTION = """
The following are files uploaded by the user. Each file is identified by its filename, followed by its content.

{files}

Use the file contents as context when answering the user's request. When referring to information from a specific file, identify the file by its filename.
"""
