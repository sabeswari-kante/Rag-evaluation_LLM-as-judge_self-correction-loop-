"""
generic_queries.py — Short-circuit handler for greetings and generic questions.

Keeps these out of the RAG pipeline to save API calls and latency.
"""

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
        (
            "I'm your Paul Graham Essays Assistant 📚. "
            "I can answer questions about Paul Graham's essays — his ideas on startups, "
            "wealth, determination, identity, hiring, fundraising, and more. Just ask!"
        ),
    ),
    (
        ("how are you", "how r you", "you good", "you okay", "how do you do"),
        "I'm doing great, thanks for asking! 😊 Ready to help you explore Paul Graham's essays. "
        "What would you like to know?",
    ),
    (
        ("thank you", "thanks", "thx", "ty", "thank u", "many thanks", "much appreciated"),
        "You're welcome! 😊 Feel free to ask more questions about Paul Graham's ideas anytime.",
    ),
    (
        ("bye", "goodbye", "see you", "cya", "take care", "later", "see ya", "farewell"),
        "Goodbye! 👋 Come back anytime you want to explore Paul Graham's essays!",
    ),
    (
        ("help", "help me", "i need help", "can you help"),
        (
            "Of course! 🙌 I can help you with questions about Paul Graham's essays — "
            "topics like startups, wealth creation, determination, identity, hiring, "
            "fundraising, and founder advice. What's your question?"
        ),
    ),
    (
        ("what is paul graham", "who is paul graham", "tell me about paul graham"),
        (
            "Paul Graham is a programmer, essayist, and co-founder of Y Combinator, "
            "the world's most influential startup accelerator. His essays cover topics like "
            "how to start a startup, wealth creation, determination, and hacker culture. "
            "Ask me anything about his ideas! 🖊️"
        ),
    ),
    (
        ("what essays do you know", "which essays", "list essays", "what topics"),
        (
            "I have knowledge of several Paul Graham essays including:\n"
            "- *Do Things That Don't Scale*\n"
            "- *How to Get Startup Ideas*\n"
            "- *Keep Your Identity Small*\n"
            "- *How to Make Wealth*\n"
            "- *Hackers and Painters*\n"
            "- *The Anatomy of Determination*\n"
            "- *Before the Startup*\n"
            "- *How to Raise Money*\n"
            "- *Default Alive or Default Dead*\n"
            "- *Mean People Fail*\n"
            "...and more! Ask me anything about these topics."
        ),
    ),
    (
        ("ok", "okay", "got it", "i see", "understood", "noted", "alright", "sure"),
        "Great! 😊 Let me know if you have any questions about Paul Graham's essays.",
    ),
    (
        ("cool", "awesome", "nice", "great", "wow", "amazing", "impressive"),
        "Glad you think so! 😄 Ask me anything about Paul Graham's essays anytime.",
    ),
]


def handle_generic_query(query: str) -> Optional[str]:
    """
    Checks if the query is a greeting or generic question.

    Returns:
        A direct string reply if it matches a generic pattern.
        None if it's a real question → should go through the RAG pipeline.
    """
    q = query.strip().lower()

    # check greetings
    for greeting in GREETINGS:
        if q == greeting or q.startswith(greeting + " ") or q.startswith(greeting + ","):
            return (
                "Hey there! 👋 I'm your Paul Graham Essays Assistant. "
                "Ask me anything about his ideas on startups, wealth, determination, "
                "identity, hiring, fundraising, and more!"
            )

    # check generic patterns
    for patterns, reply in GENERIC_PATTERNS:
        if any(pattern in q for pattern in patterns):
            return reply

    # not a generic query — pass to RAG pipeline
    return None