from unittest.mock import MagicMock

from src.services.chat_service import ChatService
from src.models.chat import Conversation


# Mock LLM provider
provider = MagicMock()

provider.provider_name = "test-provider"

provider.chat.return_value = iter(
    [
        "Based on the documents, ",
        "Python is a programming language."
    ]
)


# Mock RAG service
rag_service = MagicMock()

rag_result = MagicMock()
rag_result.context = """
Python is a high-level programming language.
It is widely used for AI, automation, and web development.
"""

rag_service.query.return_value = rag_result


# Create ChatService with RAG enabled
chat_service = ChatService(
    provider=provider,
    model="test-model",
    rag_service=rag_service
)


conversation = Conversation()


response = chat_service.send_message(
    conversation,
    "What is Python?"
)


print("\nAssistant Response:")
print(response)


print("\nUser message stored in conversation:")
print(conversation.messages[0].content)


print("\nRAG called:")
rag_service.query.assert_called_once_with(
    "What is Python?"
)

print("\nRAG integration test passed ✅")