"""Structured memory using knowledge graph."""

from typing import Dict, Any, Optional
from loguru import logger
from graph.neo4j_client import Neo4jClient


class StructuredMemory:
    """Structured memory for user data and preferences in graph."""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize structured memory.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j_client = neo4j_client
        logger.info("Initialized StructuredMemory")

    async def store_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
    ) -> None:
        """
        Store user preference.

        Args:
            user_id: User identifier
            preference_key: Preference key
            preference_value: Preference value
        """
        query = """
        MERGE (u:User {id: $user_id})
        SET u[$key] = $value
        """
        await self.neo4j_client.execute_query(
            query,
            {
                "user_id": user_id,
                "key": preference_key,
                "value": preference_value,
            },
        )
        logger.debug(f"Stored user preference: {user_id}.{preference_key}")

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get all user preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences dictionary
        """
        query = """
        MATCH (u:User {id: $user_id})
        RETURN properties(u) as preferences
        """
        result = await self.neo4j_client.execute_query(query, {"user_id": user_id})

        if result:
            return result[0].get("preferences", {})

        return {}

    async def get_user_context(self, user_id: str) -> str:
        """
        Get formatted user context for LLM.

        Args:
            user_id: User identifier

        Returns:
            Formatted context string
        """
        preferences = await self.get_user_preferences(user_id)

        if not preferences:
            return ""

        context_parts = [f"User context for {user_id}:"]
        for key, value in preferences.items():
            if key != "id":
                context_parts.append(f"- {key}: {value}")

        return "\n".join(context_parts)
