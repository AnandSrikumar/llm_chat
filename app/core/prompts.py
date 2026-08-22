SYSTEM_PROMPT = """
You are a useful assistant, Answer the user clearly. 
"""

NAME_GENERATOR_PROMPT = """
You are a conversation summarizer. You will generate a one word name for the conversation. Keep the name to 50 characters
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
