"""Graph builder for constructing knowledge graphs from documents."""

from typing import List
from loguru import logger
from core.models import Document, DocumentChunk
from graph.neo4j_client import Neo4jClient
from graph.entity_extractor import EntityExtractor


class GraphBuilder:
    """Build knowledge graph from documents."""

    def __init__(
        self,
        neo4j_client: Neo4jClient = None,
        entity_extractor: EntityExtractor = None,
    ):
        """
        Initialize graph builder.

        Args:
            neo4j_client: Neo4j client instance
            entity_extractor: Entity extractor instance
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.entity_extractor = entity_extractor or EntityExtractor()

        logger.info("Initialized GraphBuilder")

    async def build_from_document(self, document: Document) -> None:
        """
        Build graph from a single document.

        Args:
            document: Document to process
        """
        logger.info(f"Building graph from document: {document.id}")

        # Extract entities and relations
        entities, relations = self.entity_extractor.extract_entities_and_relations(
            text=document.content,
            doc_id=document.id,
        )

        # Store in Neo4j
        await self.neo4j_client.connect()

        if entities:
            await self.neo4j_client.create_entities_batch(entities)

        if relations:
            await self.neo4j_client.create_relations_batch(relations)

        logger.info(
            f"Created {len(entities)} entities and {len(relations)} relations "
            f"for document {document.id}"
        )

    async def build_from_documents(self, documents: List[Document]) -> None:
        """
        Build graph from multiple documents.

        Args:
            documents: List of documents
        """
        logger.info(f"Building graph from {len(documents)} documents")

        for doc in documents:
            try:
                await self.build_from_document(doc)
            except Exception as e:
                logger.error(f"Failed to process document {doc.id}: {e}")

        logger.info("Graph building completed")

    async def build_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Build graph from document chunks.

        Args:
            chunks: List of document chunks
        """
        logger.info(f"Building graph from {len(chunks)} chunks")

        all_entities = []
        all_relations = []

        # Extract from all chunks
        for chunk in chunks:
            try:
                entities, relations = self.entity_extractor.extract_entities_and_relations(
                    text=chunk.content,
                    doc_id=chunk.document_id,
                )
                all_entities.extend(entities)
                all_relations.extend(relations)

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")

        # Batch insert to Neo4j
        await self.neo4j_client.connect()

        if all_entities:
            await self.neo4j_client.create_entities_batch(all_entities)

        if all_relations:
            await self.neo4j_client.create_relations_batch(relations)

        logger.info(
            f"Created {len(all_entities)} entities and {len(all_relations)} relations "
            f"from {len(chunks)} chunks"
        )

    async def enrich_with_embeddings(self, entity_ids: List[str], embeddings: List[List[float]]):
        """
        Add embeddings to entities (for hybrid search).

        Args:
            entity_ids: List of entity IDs
            embeddings: Corresponding embeddings
        """
        query = """
        UNWIND $data as item
        MATCH (e:Entity {id: item.id})
        SET e.embedding = item.embedding
        """

        data = [{"id": id, "embedding": emb} for id, emb in zip(entity_ids, embeddings)]

        await self.neo4j_client.execute_query(query, {"data": data})
        logger.info(f"Enriched {len(entity_ids)} entities with embeddings")

    async def close(self) -> None:
        """Close connections."""
        await self.neo4j_client.close()
