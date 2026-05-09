"""Entity and relation extraction from text."""

import uuid
from typing import List, Tuple, Dict, Any
import spacy
from loguru import logger
from core.models import Entity, Relation
from core.types import EntityType, RelationType
from core.exceptions import EntityExtractionError


class EntityExtractor:
    """Extract entities and relations from text using NLP."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize entity extractor.

        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model {model_name} not found. "
                "Download with: python -m spacy download en_core_web_sm"
            )
            raise EntityExtractionError(f"spaCy model {model_name} not available")

    def extract_entities(self, text: str, doc_id: str = None) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text
            doc_id: Optional document ID for tracking

        Returns:
            List of extracted entities
        """
        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                # Map spaCy entity types to our EntityType enum
                entity_type = self._map_entity_type(ent.label_)

                entity = Entity(
                    id=str(uuid.uuid4()),
                    name=ent.text,
                    type=entity_type,
                    properties={
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "document_id": doc_id,
                    },
                )
                entities.append(entity)

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise EntityExtractionError(f"Extraction failed: {str(e)}")

    def extract_relations(
        self,
        text: str,
        entities: List[Entity] = None,
    ) -> List[Relation]:
        """
        Extract relations between entities.

        Args:
            text: Input text
            entities: Pre-extracted entities (optional)

        Returns:
            List of extracted relations
        """
        try:
            doc = self.nlp(text)

            # If entities not provided, extract them
            if entities is None:
                entities = self.extract_entities(text)

            relations = []

            # Simple relation extraction based on dependency parsing
            for token in doc:
                if token.dep_ in ["nsubj", "dobj", "pobj"]:
                    # Find subject-verb-object patterns
                    subject = token
                    verb = token.head

                    # Look for object
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj"] and child != subject:
                            object_token = child

                            # Try to match with extracted entities
                            source_entity = self._find_entity_by_text(subject.text, entities)
                            target_entity = self._find_entity_by_text(object_token.text, entities)

                            if source_entity and target_entity:
                                relation_type = self._infer_relation_type(verb.lemma_)

                                relation = Relation(
                                    id=str(uuid.uuid4()),
                                    source_id=source_entity.id,
                                    target_id=target_entity.id,
                                    type=relation_type,
                                    properties={
                                        "verb": verb.lemma_,
                                        "text": f"{subject.text} {verb.text} {object_token.text}",
                                    },
                                    weight=1.0,
                                )
                                relations.append(relation)

            logger.info(f"Extracted {len(relations)} relations")
            return relations

        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            raise EntityExtractionError(f"Relation extraction failed: {str(e)}")

    def extract_entities_and_relations(
        self,
        text: str,
        doc_id: str = None,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract both entities and relations from text.

        Args:
            text: Input text
            doc_id: Optional document ID

        Returns:
            Tuple of (entities, relations)
        """
        entities = self.extract_entities(text, doc_id=doc_id)
        relations = self.extract_relations(text, entities=entities)

        return entities, relations

    @staticmethod
    def _map_entity_type(spacy_label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
        }

        return mapping.get(spacy_label, EntityType.CONCEPT)

    @staticmethod
    def _infer_relation_type(verb: str) -> RelationType:
        """Infer relation type from verb."""
        verb = verb.lower()

        if verb in ["create", "make", "build", "develop"]:
            return RelationType.CREATED_BY
        elif verb in ["work", "employ"]:
            return RelationType.WORKS_FOR
        elif verb in ["locate", "base", "situate"]:
            return RelationType.LOCATED_IN
        elif verb in ["mention", "discuss", "describe"]:
            return RelationType.MENTIONS
        elif verb in ["contain", "include"]:
            return RelationType.PART_OF
        elif verb in ["occur", "happen"]:
            return RelationType.OCCURRED_AT
        else:
            return RelationType.RELATES_TO

    @staticmethod
    def _find_entity_by_text(text: str, entities: List[Entity]) -> Entity:
        """Find entity by matching text."""
        text = text.lower().strip()
        for entity in entities:
            if entity.name.lower().strip() == text or text in entity.name.lower():
                return entity
        return None
