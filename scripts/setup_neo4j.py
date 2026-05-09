"""Setup script for Neo4j knowledge graph."""

import asyncio
from loguru import logger
from graph.neo4j_client import Neo4jClient


async def setup_neo4j():
    """Initialize Neo4j database with schema and indexes."""
    logger.info("Setting up Neo4j database...")

    client = Neo4jClient()

    try:
        # Connect
        await client.connect()
        logger.info("Connected to Neo4j")

        # Create indexes
        await client.create_indexes()
        logger.info("Created indexes")

        # Create constraints
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                await client.execute_query(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint creation skipped: {e}")

        logger.info("Neo4j setup completed successfully")

    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
        raise

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(setup_neo4j())
