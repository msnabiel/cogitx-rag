"""Neo4j database client for knowledge graph operations."""

from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from loguru import logger
from core.models import Entity, Relation
from core.exceptions import GraphDatabaseError
from config.settings import settings


class Neo4jClient:
    """Neo4j client for graph database operations."""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Database user
            password: Database password
            database: Database name
        """
        self.uri = uri or settings.graph.neo4j_uri
        self.user = user or settings.graph.neo4j_user
        self.password = password or settings.graph.neo4j_password
        self.database = database or settings.graph.neo4j_database

        self.driver: Optional[AsyncDriver] = None
        logger.info(f"Initialized Neo4j client for {self.uri}")

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise GraphDatabaseError(f"Connection failed: {str(e)}")

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Closed Neo4j connection")

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """
        Execute Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results as list of dictionaries
        """
        if not self.driver:
            await self.connect()

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise GraphDatabaseError(f"Query failed: {str(e)}")

    async def create_entity(self, entity: Entity) -> str:
        """
        Create entity node in graph.

        Args:
            entity: Entity model

        Returns:
            Entity ID
        """
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.properties = $properties
        RETURN e.id as id
        """
        parameters = {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value,
            "properties": entity.properties,
        }

        result = await self.execute_query(query, parameters)
        return result[0]["id"] if result else entity.id

    async def create_entities_batch(self, entities: List[Entity]) -> None:
        """
        Create multiple entities in batch.

        Args:
            entities: List of entities
        """
        query = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        SET e.name = entity.name,
            e.type = entity.type,
            e.properties = entity.properties
        """
        parameters = {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type.value,
                    "properties": e.properties,
                }
                for e in entities
            ]
        }

        await self.execute_query(query, parameters)
        logger.info(f"Created {len(entities)} entities")

    async def create_relation(self, relation: Relation) -> str:
        """
        Create relationship between entities.

        Args:
            relation: Relation model

        Returns:
            Relation ID
        """
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES {id: $id, type: $type}]->(target)
        SET r.properties = $properties,
            r.weight = $weight
        RETURN r.id as id
        """
        parameters = {
            "id": relation.id,
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "type": relation.type.value,
            "properties": relation.properties,
            "weight": relation.weight,
        }

        result = await self.execute_query(query, parameters)
        return result[0]["id"] if result else relation.id

    async def create_relations_batch(self, relations: List[Relation]) -> None:
        """
        Create multiple relations in batch.

        Args:
            relations: List of relations
        """
        query = """
        UNWIND $relations as rel
        MATCH (source:Entity {id: rel.source_id})
        MATCH (target:Entity {id: rel.target_id})
        MERGE (source)-[r:RELATES {id: rel.id, type: rel.type}]->(target)
        SET r.properties = rel.properties,
            r.weight = rel.weight
        """
        parameters = {
            "relations": [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type.value,
                    "properties": r.properties,
                    "weight": r.weight,
                }
                for r in relations
            ]
        }

        await self.execute_query(query, parameters)
        logger.info(f"Created {len(relations)} relations")

    async def find_entity_by_name(self, name: str, limit: int = 10) -> List[Entity]:
        """
        Find entities by name.

        Args:
            name: Entity name to search
            limit: Maximum results

        Returns:
            List of entities
        """
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e
        LIMIT $limit
        """
        result = await self.execute_query(query, {"name": name, "limit": limit})

        entities = []
        for record in result:
            entity_data = record["e"]
            entities.append(
                Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    type=entity_data["type"],
                    properties=entity_data.get("properties", {}),
                )
            )

        return entities

    async def traverse_graph(
        self,
        start_entity_id: str,
        max_depth: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Traverse graph from starting entity.

        Args:
            start_entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            relation_types: Filter by relation types

        Returns:
            Subgraph with entities and relations
        """
        rel_filter = ""
        if relation_types:
            types_str = "|".join(relation_types)
            rel_filter = f":{types_str}"

        query = f"""
        MATCH path = (start:Entity {{id: $start_id}})-[r{rel_filter}*1..{max_depth}]-(connected:Entity)
        RETURN start, connected, relationships(path) as rels
        LIMIT 100
        """

        result = await self.execute_query(query, {"start_id": start_entity_id})

        entities = {}
        relations = []

        for record in result:
            # Process start entity
            start_data = record["start"]
            if start_data["id"] not in entities:
                entities[start_data["id"]] = start_data

            # Process connected entity
            connected_data = record["connected"]
            if connected_data["id"] not in entities:
                entities[connected_data["id"]] = connected_data

            # Process relations
            for rel in record["rels"]:
                relations.append(rel)

        return {"entities": list(entities.values()), "relations": relations}

    async def create_indexes(self) -> None:
        """Create database indexes for performance."""
        queries = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]

        for query in queries:
            try:
                await self.execute_query(query)
                logger.info(f"Created index: {query}")
            except Exception as e:
                logger.warning(f"Index creation skipped: {e}")
