import asyncio
import time
from pathlib import Path
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

load_dotenv()

async def test_structured_vs_traditional():
    file_path = Path("examples/data/AA21-287A.pdf")
    
    # Load and split PDF
    loader = PdfLoader()
    document = await loader.run(filepath=file_path)
    
    splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = await splitter.run(document.text)
    
    print(f"Document loaded: {len(document.text)} characters")
    print(f"Chunks created: {len(chunks.chunks)} chunks")
    
    # Test with JSON object mode
    print("\n=== JSON Object Mode ===")
    extractor_traditional = LLMEntityRelationExtractor(structured_output=False)
    
    start_time = time.time()
    result_traditional = await extractor_traditional.run(chunks)
    traditional_time = time.time() - start_time
    
    print(f"Execution time: {traditional_time:.2f} seconds")
    print(f"Nodes extracted: {len(result_traditional.nodes)}")
    print(f"Relationships extracted: {len(result_traditional.relationships)}")
    
    # Test with structured outputs
    print("\n=== Structured Outputs ===")
    extractor_structured = LLMEntityRelationExtractor(structured_output=True)
    
    start_time = time.time()
    result_structured = await extractor_structured.run(chunks)
    structured_time = time.time() - start_time
    
    print(f"Execution time: {structured_time:.2f} seconds")
    print(f"Nodes extracted: {len(result_structured.nodes)}")
    print(f"Relationships extracted: {len(result_structured.relationships)}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    time_diff = structured_time - traditional_time
    time_percentage = (time_diff / traditional_time) * 100
    
    print(f"Time difference: {time_diff:.2f} seconds")
    print(f"Structured outputs performance: {abs(time_percentage):.1f}% {'slower' if time_diff > 0 else 'faster'}")
    
    # Extraction comparison
    node_diff = len(result_structured.nodes) - len(result_traditional.nodes)
    rel_diff = len(result_structured.relationships) - len(result_traditional.relationships)
    
    print(f"\n=== Extraction Comparison ===")
    print(f"Node difference: {node_diff:+d}")
    print(f"Relationship difference: {rel_diff:+d}")
    print(f"Extraction: Found {node_diff:+d} nodes and {rel_diff:+d} relationships vs traditional approach")
    
    # Calculate extraction rates
    traditional_total = len(result_traditional.nodes) + len(result_traditional.relationships)
    structured_total = len(result_structured.nodes) + len(result_structured.relationships)
    extraction_diff = structured_total - traditional_total
    
    extraction_percentage = (extraction_diff / traditional_total) * 100
    
    print(f"Total entities/relationships: {traditional_total} (traditional) vs {structured_total} (structured) = {extraction_diff:+d} difference")
    print(f"Structured outputs extraction: {abs(extraction_percentage):.1f}% {'more' if extraction_diff > 0 else 'fewer'} entities/relationships")

if __name__ == "__main__":
    asyncio.run(test_structured_vs_traditional())
