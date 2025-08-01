.. _api-documentation:

API Documentation
#################

.. _components-section:

**********
Components
**********

Component
=========

.. autoclass:: neo4j_graphrag.experimental.pipeline.component.Component
  :members: run, run_with_context

DataLoader
==========

.. autoclass:: neo4j_graphrag.experimental.components.pdf_loader.DataLoader
    :members: run, get_document_metadata

PdfLoader
=========

.. autoclass:: neo4j_graphrag.experimental.components.pdf_loader.PdfLoader
    :members: run, load_file

TextSplitter
============

.. autoclass:: neo4j_graphrag.experimental.components.text_splitters.base.TextSplitter
    :members: run

FixedSizeSplitter
=================

.. autoclass:: neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter
    :members: run

LangChainTextSplitterAdapter
============================

.. autoclass:: neo4j_graphrag.experimental.components.text_splitters.langchain.LangChainTextSplitterAdapter
    :members: run

LlamaIndexTextSplitterAdapter
=============================

.. autoclass:: neo4j_graphrag.experimental.components.text_splitters.llamaindex.LlamaIndexTextSplitterAdapter
    :members: run

TextChunkEmbedder
=================

.. autoclass:: neo4j_graphrag.experimental.components.embedder.TextChunkEmbedder
    :members: run

LexicalGraphBuilder
===================

.. autoclass:: neo4j_graphrag.experimental.components.lexical_graph.LexicalGraphBuilder
    :members:
    :exclude-members: component_inputs, component_outputs

Neo4jChunkReader
================

.. autoclass:: neo4j_graphrag.experimental.components.neo4j_reader.Neo4jChunkReader
    :members:
    :exclude-members: component_inputs, component_outputs

SchemaBuilder
=============

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaBuilder
    :members: run

SchemaFromTextExtractor
-----------------------

.. autoclass:: neo4j_graphrag.experimental.components.schema.SchemaFromTextExtractor
    :members: run

EntityRelationExtractor
=======================

.. autoclass:: neo4j_graphrag.experimental.components.entity_relation_extractor.EntityRelationExtractor
    :members:
    :exclude-members: component_inputs, component_outputs

LLMEntityRelationExtractor
==========================

.. autoclass:: neo4j_graphrag.experimental.components.entity_relation_extractor.LLMEntityRelationExtractor
    :members: run

KGWriter
========

.. autoclass:: neo4j_graphrag.experimental.components.kg_writer.KGWriter
    :members: run

Neo4jWriter
===========

.. autoclass:: neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter
    :members: run

SinglePropertyExactMatchResolver
================================

.. autoclass:: neo4j_graphrag.experimental.components.resolver.SinglePropertyExactMatchResolver
    :members: run

SpaCySemanticMatchResolver
==========================

.. autoclass:: neo4j_graphrag.experimental.components.resolver.SpaCySemanticMatchResolver
    :members: run

FuzzyMatchResolver
==================

.. autoclass:: neo4j_graphrag.experimental.components.resolver.FuzzyMatchResolver
    :members: run

.. _pipeline-section:

*********
Pipelines
*********

Pipeline
========

.. autoclass:: neo4j_graphrag.experimental.pipeline.Pipeline
    :members: run, add_component, connect, draw

SimpleKGPipeline
================

.. autoclass:: neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline
    :members: run_async


************
Config files
************


SimpleKGPipelineConfig
======================

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder.SimpleKGPipelineConfig


PipelineRunner
==============

.. autoclass:: neo4j_graphrag.experimental.pipeline.config.runner.PipelineRunner


.. _retrievers-section:

**********
Retrievers
**********

RetrieverInterface
==================

.. autoclass:: neo4j_graphrag.retrievers.base.Retriever
    :members:


VectorRetriever
===============

.. autoclass:: neo4j_graphrag.retrievers.VectorRetriever
    :members: search

VectorCypherRetriever
=====================

.. autoclass:: neo4j_graphrag.retrievers.VectorCypherRetriever
    :members: search


HybridRetriever
===============

.. autoclass:: neo4j_graphrag.retrievers.HybridRetriever
    :members: search


HybridCypherRetriever
=====================

.. autoclass:: neo4j_graphrag.retrievers.HybridCypherRetriever
    :members: search

Text2CypherRetriever
=====================

.. autoclass:: neo4j_graphrag.retrievers.Text2CypherRetriever
    :members: search


*******************
External Retrievers
*******************

This section includes retrievers that integrate with databases external to Neo4j.


WeaviateNeo4jRetriever
======================

.. autoclass:: neo4j_graphrag.retrievers.external.weaviate.weaviate.WeaviateNeo4jRetriever
    :members: search


PineconeNeo4jRetriever
======================

.. autoclass:: neo4j_graphrag.retrievers.external.pinecone.pinecone.PineconeNeo4jRetriever
    :members: search

QdrantNeo4jRetriever
====================

.. autoclass:: neo4j_graphrag.retrievers.external.qdrant.qdrant.QdrantNeo4jRetriever
    :members: search


********
Embedder
********

.. autoclass:: neo4j_graphrag.embeddings.base.Embedder
    :members:

SentenceTransformerEmbeddings
=============================

.. autoclass:: neo4j_graphrag.embeddings.sentence_transformers.SentenceTransformerEmbeddings
    :members:

OpenAIEmbeddings
================

.. autoclass:: neo4j_graphrag.embeddings.openai.OpenAIEmbeddings
    :members:

AzureOpenAIEmbeddings
=====================

.. autoclass:: neo4j_graphrag.embeddings.openai.AzureOpenAIEmbeddings
    :members:

OllamaEmbeddings
================

.. autoclass:: neo4j_graphrag.embeddings.ollama.OllamaEmbeddings
    :members:

VertexAIEmbeddings
==================

.. autoclass:: neo4j_graphrag.embeddings.vertexai.VertexAIEmbeddings
    :members:

MistralAIEmbeddings
===================

.. autoclass:: neo4j_graphrag.embeddings.mistral.MistralAIEmbeddings
    :members:

CohereEmbeddings
================

.. autoclass:: neo4j_graphrag.embeddings.cohere.CohereEmbeddings
    :members:

**********
Generation
**********

LLM
===

LLMInterface
------------

.. autoclass:: neo4j_graphrag.llm.LLMInterface
    :members:


OpenAILLM
---------

.. autoclass:: neo4j_graphrag.llm.openai_llm.OpenAILLM
    :members:
    :undoc-members: get_messages, client_class, async_client_class


AzureOpenAILLM
--------------

.. autoclass:: neo4j_graphrag.llm.openai_llm.AzureOpenAILLM
    :members:
    :undoc-members: get_messages, client_class, async_client_class

OllamaLLM
---------

.. autoclass:: neo4j_graphrag.llm.ollama_llm.OllamaLLM
    :members:


VertexAILLM
-----------

.. autoclass:: neo4j_graphrag.llm.vertexai_llm.VertexAILLM
    :members:

AnthropicLLM
------------

.. autoclass:: neo4j_graphrag.llm.anthropic_llm.AnthropicLLM
    :members:


CohereLLM
---------

.. autoclass:: neo4j_graphrag.llm.cohere_llm.CohereLLM
    :members:


MistralAILLM
------------

.. autoclass:: neo4j_graphrag.llm.mistralai_llm.MistralAILLM
    :members:


Rate Limiting
=============

RateLimitHandler
----------------

.. autoclass:: neo4j_graphrag.llm.rate_limit.RateLimitHandler
    :members:

RetryRateLimitHandler
---------------------

.. autoclass:: neo4j_graphrag.llm.rate_limit.RetryRateLimitHandler
    :members:

NoOpRateLimitHandler
--------------------

.. autoclass:: neo4j_graphrag.llm.rate_limit.NoOpRateLimitHandler
    :members:


PromptTemplate
==============

.. autoclass:: neo4j_graphrag.generation.prompts.PromptTemplate
    :members:


RagTemplate
-----------

.. autoclass:: neo4j_graphrag.generation.prompts.RagTemplate
    :members:
    :exclude-members: format

ERExtractionTemplate
--------------------

.. autoclass:: neo4j_graphrag.generation.prompts.ERExtractionTemplate
    :members:
    :exclude-members: format

SchemaExtractionTemplate
------------------------

.. autoclass:: neo4j_graphrag.generation.prompts.SchemaExtractionTemplate
    :members:
    :exclude-members: format

Text2CypherTemplate
--------------------

.. autoclass:: neo4j_graphrag.generation.prompts.Text2CypherTemplate
    :members:
    :exclude-members: format


****
RAG
****

GraphRAG
========

.. autoclass:: neo4j_graphrag.generation.graphrag.GraphRAG
    :members:


.. _database-interaction-section:

********************
Database Interaction
********************

.. _create-vector-index:

.. autofunction:: neo4j_graphrag.indexes.create_vector_index

.. _create-fulltext-index:

.. autofunction:: neo4j_graphrag.indexes.create_fulltext_index

.. autofunction:: neo4j_graphrag.indexes.drop_index_if_exists

.. autofunction:: neo4j_graphrag.indexes.upsert_vectors

.. autofunction:: neo4j_graphrag.indexes.upsert_vector

.. autofunction:: neo4j_graphrag.indexes.upsert_vector_on_relationship

.. autofunction:: neo4j_graphrag.indexes.async_upsert_vector

.. autofunction:: neo4j_graphrag.indexes.async_upsert_vector_on_relationship

.. autofunction:: neo4j_graphrag.indexes.retrieve_vector_index_info

.. autofunction:: neo4j_graphrag.indexes.retrieve_fulltext_index_info

.. autofunction:: neo4j_graphrag.schema.get_structured_schema

.. autofunction:: neo4j_graphrag.schema.get_schema

.. autofunction:: neo4j_graphrag.schema.format_schema


***************
Message History
***************

.. autoclass:: neo4j_graphrag.message_history.InMemoryMessageHistory

.. autoclass:: neo4j_graphrag.message_history.Neo4jMessageHistory


******
Errors
******


* :class:`neo4j_graphrag.exceptions.Neo4jGraphRagError`

  * :class:`neo4j_graphrag.exceptions.RetrieverInitializationError`

  * :class:`neo4j_graphrag.exceptions.EmbeddingsGenerationError`

  * :class:`neo4j_graphrag.exceptions.SearchValidationError`

  * :class:`neo4j_graphrag.exceptions.FilterValidationError`

  * :class:`neo4j_graphrag.exceptions.EmbeddingRequiredError`

  * :class:`neo4j_graphrag.exceptions.InvalidRetrieverResultError`

  * :class:`neo4j_graphrag.exceptions.Neo4jIndexError`

  * :class:`neo4j_graphrag.exceptions.Neo4jVersionError`

  * :class:`neo4j_graphrag.exceptions.Text2CypherRetrievalError`

  * :class:`neo4j_graphrag.exceptions.SchemaFetchError`

  * :class:`neo4j_graphrag.exceptions.RagInitializationError`

  * :class:`neo4j_graphrag.exceptions.PromptMissingInputError`

  * :class:`neo4j_graphrag.exceptions.LLMGenerationError`

    * :class:`neo4j_graphrag.exceptions.RateLimitError`

  * :class:`neo4j_graphrag.exceptions.SchemaValidationError`

  * :class:`neo4j_graphrag.exceptions.PdfLoaderError`

  * :class:`neo4j_graphrag.exceptions.PromptMissingPlaceholderError`

  * :class:`neo4j_graphrag.exceptions.InvalidHybridSearchRankerError`

  * :class:`neo4j_graphrag.exceptions.SearchQueryParseError`

  * :class:`neo4j_graphrag.experimental.pipeline.exceptions.PipelineDefinitionError`

  * :class:`neo4j_graphrag.experimental.pipeline.exceptions.PipelineMissingDependencyError`

  * :class:`neo4j_graphrag.experimental.pipeline.exceptions.PipelineStatusUpdateError`

  * :class:`neo4j_graphrag.experimental.pipeline.exceptions.InvalidJSONError`


Neo4jGraphRagError
==================

.. autoclass:: neo4j_graphrag.exceptions.Neo4jGraphRagError
   :show-inheritance:


RetrieverInitializationError
============================

.. autoclass:: neo4j_graphrag.exceptions.RetrieverInitializationError
   :show-inheritance:


SearchValidationError
=====================

.. autoclass:: neo4j_graphrag.exceptions.SearchValidationError
   :show-inheritance:


FilterValidationError
=====================

.. autoclass:: neo4j_graphrag.exceptions.FilterValidationError
   :show-inheritance:


EmbeddingsGenerationError
=========================

.. autoclass:: neo4j_graphrag.exceptions.EmbeddingsGenerationError
   :show-inheritance:


EmbeddingRequiredError
======================

.. autoclass:: neo4j_graphrag.exceptions.EmbeddingRequiredError
   :show-inheritance:


InvalidRetrieverResultError
===========================

.. autoclass:: neo4j_graphrag.exceptions.InvalidRetrieverResultError
   :show-inheritance:


Neo4jIndexError
===============

.. autoclass:: neo4j_graphrag.exceptions.Neo4jIndexError
   :show-inheritance:


Neo4jInsertionError
===================

.. autoclass:: neo4j_graphrag.exceptions.Neo4jInsertionError
   :show-inheritance:


Neo4jVersionError
=================

.. autoclass:: neo4j_graphrag.exceptions.Neo4jVersionError
   :show-inheritance:


Text2CypherRetrievalError
=========================

.. autoclass:: neo4j_graphrag.exceptions.Text2CypherRetrievalError
   :show-inheritance:


SchemaFetchError
================

.. autoclass:: neo4j_graphrag.exceptions.SchemaFetchError
   :show-inheritance:


RagInitializationError
======================

.. autoclass:: neo4j_graphrag.exceptions.RagInitializationError
   :show-inheritance:


PromptMissingInputError
=======================

.. autoclass:: neo4j_graphrag.exceptions.PromptMissingInputError
   :show-inheritance:


LLMGenerationError
==================

.. autoclass:: neo4j_graphrag.exceptions.LLMGenerationError
   :show-inheritance:


RateLimitError
==============

.. autoclass:: neo4j_graphrag.exceptions.RateLimitError
   :show-inheritance:


SchemaValidationError
=====================

.. autoclass:: neo4j_graphrag.exceptions.SchemaValidationError
   :show-inheritance:


PdfLoaderError
==============

.. autoclass:: neo4j_graphrag.exceptions.PdfLoaderError
   :show-inheritance:


PromptMissingPlaceholderError
=============================

.. autoclass:: neo4j_graphrag.exceptions.PromptMissingPlaceholderError
   :show-inheritance:


InvalidHybridSearchRankerError
==============================

.. autoclass:: neo4j_graphrag.exceptions.InvalidHybridSearchRankerError
   :show-inheritance:


SearchQueryParseError
=====================

.. autoclass:: neo4j_graphrag.exceptions.SearchQueryParseError
   :show-inheritance:


PipelineDefinitionError
=======================

.. autoclass:: neo4j_graphrag.experimental.pipeline.exceptions.PipelineDefinitionError
   :show-inheritance:


PipelineMissingDependencyError
==============================

.. autoclass:: neo4j_graphrag.experimental.pipeline.exceptions.PipelineMissingDependencyError
   :show-inheritance:


PipelineStatusUpdateError
=========================

.. autoclass:: neo4j_graphrag.experimental.pipeline.exceptions.PipelineStatusUpdateError
   :show-inheritance:


InvalidJSONError
================

.. autoclass:: neo4j_graphrag.experimental.pipeline.exceptions.InvalidJSONError
   :show-inheritance:
