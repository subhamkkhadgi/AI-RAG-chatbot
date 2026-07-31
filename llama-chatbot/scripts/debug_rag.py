"""Temporary RAG debugging script.

Directly tests whether RAG retrieval can find uploaded documents.
Does NOT modify any existing application files.
"""

import logging
import sys

from src.config import get_settings
from src.embeddings.factory import create_embedding_provider
from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import RAGService
from src.retrieval.retriever import DocumentRetriever
from src.vectorstores.factory import create_vector_store

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("debug_rag")


def main() -> None:
    """Run a direct RAG query against the uploaded documents."""
    print("=" * 60)
    print("RAG Debug Script")
    print("=" * 60)

    # 1. Load settings
    settings = get_settings()
    print(f"\nSettings loaded:")
    print(f"  embedding_provider : {settings.embedding_provider}")
    print(f"  embedding_model    : {settings.embedding_model}")
    print(f"  qdrant_host        : {settings.qdrant_host}")
    print(f"  qdrant_port        : {settings.qdrant_port}")
    print(f"  qdrant_collection  : {settings.qdrant_collection}")

    # 2. Create embedding provider
    print(f"\nCreating embedding provider '{settings.embedding_provider}'...")
    embedding_provider = create_embedding_provider(
        settings.embedding_provider, settings
    )
    print(f"  Provider: {embedding_provider.provider_name}")

    # 3. Create vector store
    print(f"\nCreating vector store 'qdrant'...")
    vector_store = create_vector_store("qdrant", settings)
    print(f"  Provider: {vector_store.provider_name}")
    print(f"  Collection: {vector_store.collection_name}")

    # 4. Create retriever
    print(f"\nCreating DocumentRetriever...")
    retriever = DocumentRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_limit=10,
    )
    print(f"  default_limit: {retriever.default_limit}")

    # 5. Create context builder
    print(f"\nCreating ContextBuilder...")
    context_builder = ContextBuilder()
    print(f"  max_context_length: {context_builder.max_context_length}")

    # 6. Create RAG service
    print(f"\nCreating RAGService...")
    rag_service = RAGService(
        retriever=retriever,
        context_builder=context_builder,
    )
    print(f"  Retriever: {type(rag_service.retriever).__name__}")
    print(f"  ContextBuilder: {type(rag_service.context_builder).__name__}")

    # 7. Run query
    query = "Who is the CEO of the company?"
    print(f"\n{'=' * 60}")
    print(f"Executing RAG query...")
    print(f"  Query: '{query}'")
    print(f"{'=' * 60}\n")

    try:
        result = rag_service.query(query)

        # 8. Print results
        print(f"RAGResult:")
        print(f"  Query          : {result.query}")
        print(f"  Chunks retrieved: {result.retrieval_result.total_results}")
        print(f"  Context length  : {len(result.context)} chars")

        print(f"\nRetrieved chunks:")
        if result.retrieval_result.chunks:
            for i, chunk in enumerate(result.retrieval_result.chunks):
                print(f"\n  --- Chunk {i + 1} ---")
                print(f"  Chunk ID      : {chunk.chunk_id}")
                print(f"  Document ID   : {chunk.document_id}")
                print(f"  Filename      : {chunk.filename}")
                print(f"  Chunk index   : {chunk.chunk_index}")
                print(f"  Score         : {chunk.score:.6f}")
                print(f"  Page number   : {chunk.page_number}")
                print(f"  Created at    : {chunk.created_at}")
                print(f"  Text preview  : {chunk.text[:200]}...")
        else:
            print("  (no chunks retrieved)")

        print(f"\nGenerated context:")
        if result.context:
            print(f"  {result.context[:500]}")
        else:
            print("  (empty context - no context was generated)")

    except Exception as exc:
        print(f"\nERROR: RAG query failed with exception:")
        print(f"  {type(exc).__name__}: {exc}")
        logger.error("RAG query failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
