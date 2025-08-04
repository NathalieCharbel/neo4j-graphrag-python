#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import asyncio
import enum
import json
import logging
from typing import Any, List, Optional, Union

import json_repair
from openai import AsyncOpenAI
from pydantic import ValidationError, validate_call

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    Neo4jGraph,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.experimental.pipeline.exceptions import InvalidJSONError
from neo4j_graphrag.generation.prompts import ERExtractionTemplate, PromptTemplate
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.utils.logging import prettify

logger = logging.getLogger(__name__)

# JSON Schema for OpenAI structured outputs equivalent to Neo4jGraph
NEO4J_GRAPH_SCHEMA = {
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "description": "A Neo4j node",
        "properties": {
          "id": { "type": "string" },
          "label": { "type": "string" },
          "properties": {
            "type": "object",
            "description": "Node property key-value pairs",
            "additionalProperties": {
              "anyOf": [
                { "type": "boolean" },
                { "type": "integer" },
                { "type": "number" },
                { "type": "string" },
                { "type": "string", "format": "date" },
                { "type": "string", "format": "time" },
                { "type": "string", "format": "date-time" },
                { "type": "string", "pattern": "^P.*" },
                {
                  "type": "array",
                  "items": {
                    "anyOf": [
                      { "type": "boolean" },
                      { "type": "integer" },
                      { "type": "number" },
                      { "type": "string" }
                    ]
                  }
                },
                {
                  "type": "object",
                  "required": ["latitude", "longitude", "height"],
                  "properties": {
                    "latitude": { "type": "number" },
                    "longitude": { "type": "number" },
                    "height": { "type": "number" }
                  },
                  "additionalProperties": False
                }
              ]
            }
          },
          "embedding_properties": {
            "type": "object",
            "additionalProperties": {
              "type": "array",
              "items": { "type": "number" }
            }
          }
        },
        "required": ["id", "label"],
        "additionalProperties": False
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "description": "A Neo4j relationship",
        "properties": {
          "start_node_id": { "type": "string" },
          "end_node_id": { "type": "string" },
          "type": { "type": "string" },
          "properties": {
            "type": "object",
            "description": "Relationship property key-value pairs",
            "additionalProperties": {
              "anyOf": [
                { "type": "boolean" },
                { "type": "integer" },
                { "type": "number" },
                { "type": "string" },
                { "type": "string", "format": "date" },
                { "type": "string", "format": "time" },
                { "type": "string", "format": "date-time" },
                { "type": "string", "pattern": "^P.*" },
                {
                  "type": "array",
                  "items": {
                    "anyOf": [
                      { "type": "boolean" },
                      { "type": "integer" },
                      { "type": "number" },
                      { "type": "string" }
                    ]
                  }
                },
                {
                  "type": "object",
                  "required": ["latitude", "longitude", "height"],
                  "properties": {
                    "latitude": { "type": "number" },
                    "longitude": { "type": "number" },
                    "height": { "type": "number" }
                  },
                  "additionalProperties": False
                }
              ]
            }
          },
          "embedding_properties": {
            "type": "object",
            "additionalProperties": {
              "type": "array",
              "items": { "type": "number" }
            }
          }
        },
        "required": ["start_node_id", "end_node_id", "type"],
        "additionalProperties": False
      }
    }
  },
  "required": ["nodes", "relationships"],
  "additionalProperties": False
}


class OnError(enum.Enum):
    RAISE = "RAISE"
    IGNORE = "IGNORE"

    @classmethod
    def possible_values(cls) -> List[str]:
        return [e.value for e in cls]


def balance_curly_braces(json_string: str) -> str:
    """
    Balances curly braces `{}` in a JSON string. This function ensures that every opening brace has a corresponding
    closing brace, but only when they are not part of a string value. If there are unbalanced closing braces,
    they are ignored. If there are missing closing braces, they are appended at the end of the string.

    Args:
        json_string (str): A potentially malformed JSON string with unbalanced curly braces.

    Returns:
        str: A JSON string with balanced curly braces.
    """
    stack = []
    fixed_json = []
    in_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            fixed_json.append(char)
            continue
        else:
            escape = False

        if not in_string:
            if char == "{":
                stack.append(char)
                fixed_json.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
                fixed_json.append(char)
            elif char == "}" and (not stack or stack[-1] != "{"):
                continue
            else:
                fixed_json.append(char)
        else:
            fixed_json.append(char)

    # If stack is not empty, add missing closing braces
    while stack:
        stack.pop()
        fixed_json.append("}")

    return "".join(fixed_json)


def fix_invalid_json(raw_json: str) -> str:
    repaired_json = json_repair.repair_json(raw_json)
    repaired_json = repaired_json.strip()

    if repaired_json == '""':
        raise InvalidJSONError("JSON repair resulted in an empty or invalid JSON.")
    if not repaired_json:
        raise InvalidJSONError("JSON repair resulted in an empty string.")
    return repaired_json


class EntityRelationExtractor(Component):
    """Abstract class for entity relation extraction components.

    Args:
        on_error (OnError): What to do when an error occurs during extraction. Defaults to raising an error.
        create_lexical_graph (bool): Whether to include the text chunks in the graph in addition to the extracted entities and relations. Defaults to True.
    """

    def __init__(
        self,
        *args: Any,
        on_error: OnError = OnError.IGNORE,
        create_lexical_graph: bool = True,
        **kwargs: Any,
    ) -> None:
        self.on_error = on_error
        self.create_lexical_graph = create_lexical_graph

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        **kwargs: Any,
    ) -> Neo4jGraph:
        raise NotImplementedError()

    def update_ids(
        self,
        graph: Neo4jGraph,
        chunk: TextChunk,
    ) -> Neo4jGraph:
        """Make node IDs unique across chunks, document and pipeline runs
        by prefixing them with a unique prefix.
        """
        prefix = chunk.chunk_id
        for node in graph.nodes:
            node.id = f"{prefix}:{node.id}"
            if node.properties is None:
                node.properties = {}
        for rel in graph.relationships:
            rel.start_node_id = f"{prefix}:{rel.start_node_id}"
            rel.end_node_id = f"{prefix}:{rel.end_node_id}"
        return graph


class LLMEntityRelationExtractor(EntityRelationExtractor):
    """
    Extracts a knowledge graph from a series of text chunks using a large language model.
    This implementation uses OpenAI's structured outputs for more reliable JSON responses.

    Args:
        llm (LLMInterface): The language model to use for extraction (kept for compatibility).
        prompt_template (ERExtractionTemplate | str): A custom prompt template to use for extraction.
        create_lexical_graph (bool): Whether to include the text chunks in the graph in addition to the extracted entities and relations. Defaults to True.
        on_error (OnError): What to do when an error occurs during extraction. Defaults to raising an error.
        max_concurrency (int): The maximum number of concurrent tasks which can be used to make requests to the LLM.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
        from neo4j_graphrag.llm import OpenAILLM
        from neo4j_graphrag.experimental.pipeline import Pipeline

        llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0, "response_format": {"type": "object"}})

        extractor = LLMEntityRelationExtractor(llm=llm)
        pipe = Pipeline()
        pipe.add_component(extractor, "extractor")

    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.RAISE,
        max_concurrency: int = 5,
        structured_output: bool = False,
    ) -> None:
        super().__init__(on_error=on_error, create_lexical_graph=create_lexical_graph)
        self.llm = llm  # kept for compatibility but not used
        self.max_concurrency = max_concurrency
        self.structured_output = structured_output
        if isinstance(prompt_template, str):
            template = PromptTemplate(prompt_template, expected_inputs=[])
        else:
            template = prompt_template
        self.prompt_template = template
        
        # Always initialize OpenAI client
        self.openai_client = AsyncOpenAI()
        self.openai_model = "gpt-4o-2024-08-06"

    async def extract_for_chunk(
        self, schema: GraphSchema, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """Run entity extraction for a given text chunk."""
        prompt = self.prompt_template.format(
            text=chunk.text,
            schema=schema.model_dump(exclude_none=True),
            examples=examples,
        )
        
        # Always use OpenAI SDK with different response formats based on structured_output parameter
        if self.structured_output:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "graph_extraction",
                    "strict": True,
                    "schema": NEO4J_GRAPH_SCHEMA
                }
            }
        else:
            response_format = {"type": "json_object"}
            
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=response_format,
            temperature=0
        )
        llm_response_content = response.choices[0].message.content
        
        try:
            llm_generated_json = fix_invalid_json(llm_response_content)
            result = json.loads(llm_generated_json)
        except (json.JSONDecodeError, InvalidJSONError) as e:
            if self.on_error == OnError.RAISE:
                raise LLMGenerationError("LLM response is not valid JSON") from e
            else:
                logger.error(
                    f"LLM response is not valid JSON for chunk_index={chunk.index}"
                )
                logger.debug(f"Invalid JSON: {llm_response_content}")
            result = {"nodes": [], "relationships": []}
        try:
            chunk_graph = Neo4jGraph.model_validate(result)
        except ValidationError as e:
            if self.on_error == OnError.RAISE:
                raise LLMGenerationError("LLM response has improper format") from e
            else:
                logger.error(
                    f"LLM response has improper format for chunk_index={chunk.index}"
                )
                logger.debug(f"Invalid JSON format: {result}")
            chunk_graph = Neo4jGraph()
        return chunk_graph

    async def post_process_chunk(
        self,
        chunk_graph: Neo4jGraph,
        chunk: TextChunk,
        lexical_graph_builder: Optional[LexicalGraphBuilder] = None,
    ) -> None:
        """Perform post-processing after entity and relation extraction:
        - Update node IDs to make them unique across chunks
        - Build the lexical graph if requested
        """
        self.update_ids(chunk_graph, chunk)
        if lexical_graph_builder:
            await lexical_graph_builder.process_chunk_extracted_entities(
                chunk_graph,
                chunk,
            )

    def combine_chunk_graphs(
        self, lexical_graph: Optional[Neo4jGraph], chunk_graphs: List[Neo4jGraph]
    ) -> Neo4jGraph:
        """Combine sub-graphs obtained for each chunk into a single Neo4jGraph object"""
        if lexical_graph:
            graph = lexical_graph.model_copy(deep=True)
        else:
            graph = Neo4jGraph()
        for chunk_graph in chunk_graphs:
            graph.nodes.extend(chunk_graph.nodes)
            graph.relationships.extend(chunk_graph.relationships)
        return graph

    async def run_for_chunk(
        self,
        sem: asyncio.Semaphore,
        chunk: TextChunk,
        schema: GraphSchema,
        examples: str,
        lexical_graph_builder: Optional[LexicalGraphBuilder] = None,
    ) -> Neo4jGraph:
        """Run extraction, validation and post processing for a single chunk"""
        async with sem:
            chunk_graph = await self.extract_for_chunk(schema, examples, chunk)
            # final_chunk_graph = self.validate_chunk(chunk_graph, schema)
            await self.post_process_chunk(
                chunk_graph,
                chunk,
                lexical_graph_builder,
            )
            return chunk_graph

    @validate_call
    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        schema: Optional[GraphSchema] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> Neo4jGraph:
        """Perform entity and relation extraction for all chunks in a list.

        Optionally, creates the "lexical graph" by adding nodes and relationships
        to represent the document and its chunks in the returned graph
        (For more details, see the :ref:`Lexical Graph Builder doc <lexical-graph-builder>` and
        the :ref:`User Guide <lexical-graph-in-er-extraction>`)

        Args:
            chunks (TextChunks): List of text chunks to extract entities and relations from.
            document_info (Optional[DocumentInfo], optional): Document the chunks are coming from. Used in the lexical graph creation step.
            lexical_graph_config (Optional[LexicalGraphConfig], optional): Lexical graph configuration to customize node labels and relationship types in the lexical graph.
            schema (GraphSchema | None): Definition of the schema to guide the LLM in its extraction.
            examples (str): Examples for few-shot learning in the prompt.
        """
        lexical_graph_builder = None
        lexical_graph = None
        if self.create_lexical_graph:
            config = lexical_graph_config or LexicalGraphConfig()
            lexical_graph_builder = LexicalGraphBuilder(config=config)
            lexical_graph_result = await lexical_graph_builder.run(
                text_chunks=chunks, document_info=document_info
            )
            lexical_graph = lexical_graph_result.graph
        elif lexical_graph_config:
            lexical_graph_builder = LexicalGraphBuilder(config=lexical_graph_config)
        schema = schema or GraphSchema(
            node_types=(),
        )
        examples = examples or ""
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self.run_for_chunk(
                sem,
                chunk,
                schema,
                examples,
                lexical_graph_builder,
            )
            for chunk in chunks.chunks
        ]
        chunk_graphs: list[Neo4jGraph] = list(await asyncio.gather(*tasks))
        graph = self.combine_chunk_graphs(lexical_graph, chunk_graphs)
        logger.debug(f"Extracted graph: {prettify(graph)}")
        return graph
