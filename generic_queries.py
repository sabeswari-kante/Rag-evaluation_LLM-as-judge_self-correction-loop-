
from typing import Optional


GREETINGS = [
    "hi", "hey", "hello", "hii", "helo", "heya", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "sup", "yo", "greetings",
]

GENERIC_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (
        ("who are you", "what are you", "what can you do",
         "your purpose", "tell me about yourself", "introduce yourself"),
        "I'm your LangChain Documentation Helper 🤖. I can answer questions about LangChain's "
        "agents, chains, tools, retrievers, memory modules, and more. Just ask!",
    ),
    (
        ("how are you", "how r you", "you good", "you okay", "how do you do"),
        "I'm doing great, thanks for asking! 😊 Ready to help you with LangChain docs. "
        "What would you like to know?",
    ),
    (
        ("thank you", "thanks", "thx", "ty", "thank u", "many thanks", "much appreciated"),
        "You're welcome! 😊 Feel free to ask more LangChain questions anytime.",
    ),
    (
        ("bye", "goodbye", "see you", "cya", "take care", "later", "see ya", "farewell"),
        "Goodbye! 👋 Come back anytime you have LangChain questions!",
    ),
    (
        ("help", "help me", "i need help", "can you help"),
        "Of course! 🙌 I can help you with anything related to LangChain documentation — "
        "agents, chains, tools, retrievers, memory, callbacks, and more. What's your question?",
    ),
    (
        ("what is langchain", "tell me about langchain", "explain langchain"),
        "LangChain is a framework for building applications powered by large language models (LLMs). "
        "It provides building blocks like chains, agents, retrievers, memory, and tools to make it "
        "easy to build production-ready LLM apps. Ask me anything specific about it! 🔗",
    ),
    (
        ("ok", "okay", "got it", "i see", "understood", "noted", "alright", "sure"),
        "Great! 😊 Let me know if you have any LangChain questions.",
    ),
    (
        ("cool", "awesome", "nice", "great", "wow", "amazing", "impressive"),
        "Glad you think so! 😄 Ask me anything about LangChain anytime.",
    ),
]


def handle_generic_query(query: str) -> Optional[str]:
    """
    Checks if the query is a greeting or generic question.

    Returns:
        A direct string reply if it matches a generic pattern.
        None if it's a real documentation query → should go through RAG pipeline.
    """
    q = query.strip().lower()

    # check greetings
    for greeting in GREETINGS:
        if q == greeting or q.startswith(greeting + " ") or q.startswith(greeting + ","):
            return (
                f"Hey there! 👋 I'm your LangChain Documentation Helper. "
                f"Ask me anything about LangChain — agents, chains, tools, "
                f"retrievers, memory, and more!"
            )

    # check generic patterns
    for patterns, reply in GENERIC_PATTERNS:
        if any(pattern in q for pattern in patterns):
            return reply

    # not a generic query
    return None