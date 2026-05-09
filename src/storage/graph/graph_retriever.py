"""Graph-based retrieval with multi-hop traversal."""

from typing import List, Optional, Dict, Any
from loguru import logger
from core.models import RetrievalResult
from core.types import RetrieverType
from graph.neo4j_client import Neo4jClient
from graph.entity_extractor import EntityExtractor
from config.settings import settings


class GraphRetriever:
    """Retrieve information from knowledge graph."""

    def __init__(
        self,
        neo4j_client: Neo4jClient = None,
        entity_extractor: EntityExtractor = None,
    ):
        """
        Initialize graph retriever.

        Args:
            neo4j_client: Neo4j client instance
            entity_extractor: Entity extractor for query understanding
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.entity_extractor = entity_extractor or EntityExtractor()

        logger.info("Initialized GraphRetriever")

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        max_depth: int = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant information from graph based on query.

        Args:
            query: Query text
            top_k: Number of results to return
            max_depth: Maximum traversal depth

        Returns:
            List of retrieval results
        """
        top_k = top_k or settings.retrieval.graph_max_results
        max_depth = max_depth or settings.retrieval.graph_max_depth

        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query)

        if not query_entities:
            logger.info("No entities found in query")
            return []

        # Search for matching entities in graph
        results = []

        await self.neo4j_client.connect()

        for query_entity in query_entities:
            # Find similar entities in graph
            graph_entities = await self.neo4j_client.find_entity_by_name(
                query_entity.name,
                limit=5,
            )

            # Traverse graph from each matched entity
            for entity in graph_entities:
                subgraph = await self.neo4j_client.traverse_graph(
                    start_entity_id=entity.id,
                    max_depth=max_depth,
                )

                # Convert subgraph to retrieval result
                result = self._subgraph_to_result(subgraph, entity)
                results.append(result)

        # Rank and limit results
        results = results[:top_k]

        logger.info(f"Graph retrieval returned {len(results)} results")
        return results

    async def multi_hop_query(
        self,
        start_entity_names: List[str],
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Perform multi-hop graph traversal query.

        Args:
            start_entity_names: Starting entity names
            relation_types: Filter by relation types
            max_depth: Maximum hops

        Returns:
            Subgraph with entities and relations
        """
        await self.neo4j_client.connect()

        all_entities = []
        all_relations = []

        for entity_name in start_entity_names:
            # Find entity
            entities = await self.neo4j_client.find_entity_by_name(entity_name, limit=1)

            if not entities:
                continue

            entity = entities[0]

            # Traverse graph
            subgraph = await self.neo4j_client.traverse_graph(
                start_entity_id=entity.id,
                max_depth=max_depth,
                relation_types=relation_types,
            )

            all_entities.extend(subgraph["entities"])
            all_relations.extend(subgraph["relations"])

        # Deduplicate
        unique_entities = {e["id"]: e for e in all_entities}

        return {
            "entities": list(unique_entities.values()),
            "relations": all_relations,
        }

    async def get_entity_context(self, entity_name: str, max_depth: int = 1) -> str:
        """
        Get textual context for an entity from the graph.

        Args:
            entity_name: Entity name
            max_depth: Traversal depth

        Returns:
            Formatted text context
        """
        subgraph = await self.multi_hop_query([entity_name], max_depth=max_depth)

        # Format as text
        context_parts = [f"Entity: {entity_name}\n"]

        for entity in subgraph["entities"]:
            context_parts.append(f"- {entity['name']} ({entity['type']})")

        if subgraph["relations"]:
            context_parts.append("\nRelationships:")
            for rel in subgraph["relations"][:10]:  # Limit relations
                context_parts.append(
                    f"- {rel.get('type', 'RELATES')}"
                )

        return "\n".join(context_parts)

    def _subgraph_to_result(self, subgraph: Dict[str, Any], start_entity) -> RetrievalResult:
        """
        Convert subgraph to retrieval result.

        Args:
            subgraph: Subgraph data
            start_entity: Starting entity

        Returns:
            Retrieval result
        """
        # Format subgraph as text content
        content_parts = [f"Starting from: {start_entity.name} ({start_entity.type.value})"]

        # Add connected entities
        if subgraph["entities"]:
            content_parts.append("\nConnected Entities:")
            for entity_data in subgraph["entities"][:10]:  # Limit to 10
                content_parts.append(f"- {entity_data['name']} ({entity_data['type']})")

        # Add relations
        if subgraph["relations"]:
            content_parts.append("\nRelationships:")
            for rel in subgraph["relations"][:10]:
                content_parts.append(f"- Connection type: {rel.get('type', 'RELATES')}")

        content = "\n".join(content_parts)

        return RetrievalResult(
            id=start_entity.id,
            content=content,
            score=1.0,  # Graph results don't have similarity scores
            source=RetrieverType.GRAPH,
            metadata={
                "entity_name": start_entity.name,
                "entity_type": start_entity.type.value,
                "num_connected_entities": len(subgraph["entities"]),
                "num_relations": len(subgraph["relations"]),
            },
        )

    async def close(self) -> None:
        """Close connections."""
        await self.neo4j_client.close()
