"""Setup script for Pinecone vector database."""

import asyncio
from loguru import logger
from vector_stores.pinecone_store import PineconeVectorStore
from config.settings import settings


async def setup_pinecone():
    """Initialize Pinecone index."""
    logger.info("Setting up Pinecone...")

    if not settings.vector_store.pinecone_api_key:
        logger.error("Pinecone API key not configured")
        return

    try:
        store = PineconeVectorStore()

        # Create index
        await store.create_index(
            metric="cosine",
            cloud="aws",
            region="us-east-1",
        )

        logger.info("Pinecone setup completed successfully")

    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_pinecone())
