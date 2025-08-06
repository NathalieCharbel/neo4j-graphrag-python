import asyncio
import time
from pathlib import Path
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

load_dotenv()

# Global metrics storage
metrics = {
    'traditional_times': [],
    'structured_times': [],
    'time_percentages': [],
    'extraction_percentages': []
}

async def test_structured_vs_traditional(file_path):
    print(f"\nTesting: {file_path.name}")
    
    loader = PdfLoader()
    document = await loader.run(filepath=file_path)
    
    splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = await splitter.run(document.text)
    
    print(f"Chunks: {len(chunks.chunks)}")
    
    extractor_traditional = LLMEntityRelationExtractor(structured_output=False)
    start_time = time.time()
    result_traditional = await extractor_traditional.run(chunks)
    traditional_time = time.time() - start_time
    
    extractor_structured = LLMEntityRelationExtractor(structured_output=True)
    start_time = time.time()
    result_structured = await extractor_structured.run(chunks)
    structured_time = time.time() - start_time
    
    time_diff = structured_time - traditional_time
    time_percentage = (time_diff / traditional_time) * 100
    
    traditional_total = len(result_traditional.nodes) + len(result_traditional.relationships)
    structured_total = len(result_structured.nodes) + len(result_structured.relationships)
    extraction_diff = structured_total - traditional_total
    extraction_percentage = (extraction_diff / traditional_total) * 100
    
    print(f"Traditional: {traditional_time:.2f}s, {traditional_total} entities")
    print(f"Structured: {structured_time:.2f}s, {structured_total} entities")
    print(f"Performance: {abs(time_percentage):.1f}% {'slower' if time_diff > 0 else 'faster'}")
    print(f"Extraction: {abs(extraction_percentage):.1f}% {'more' if extraction_diff > 0 else 'fewer'}")
    
    metrics['traditional_times'].append(traditional_time)
    metrics['structured_times'].append(structured_time)
    metrics['time_percentages'].append(time_percentage)
    metrics['extraction_percentages'].append(extraction_percentage)

async def main():
    # corpus_folder = Path("examples/data/medical_papers")
    corpus_folder = Path("examples/data/threat_intelligence")
    # corpus_folder = Path("examples/data/RILA")
    # corpus_folder = Path("examples/data/medical_reports")
    pdf_files = list(corpus_folder.glob("*.pdf"))
    
    print(f"Testing {len(pdf_files)} documents")
    
    for pdf_file in pdf_files:
        await test_structured_vs_traditional(pdf_file)
    
    print("\n=== OVERALL AVERAGES ===")
    avg_traditional_time = sum(metrics['traditional_times']) / len(metrics['traditional_times'])
    avg_structured_time = sum(metrics['structured_times']) / len(metrics['structured_times'])
    
    # Calculate performance percentage from averaged times (correct method)
    avg_time_diff = avg_structured_time - avg_traditional_time
    avg_time_percentage = (avg_time_diff / avg_traditional_time) * 100
    
    avg_extraction_percentage = sum(metrics['extraction_percentages']) / len(metrics['extraction_percentages'])
    
    print(f"Average Traditional Time: {avg_traditional_time:.2f}s")
    print(f"Average Structured Time: {avg_structured_time:.2f}s")
    print(f"Average Performance: {abs(avg_time_percentage):.1f}% {'slower' if avg_time_percentage > 0 else 'faster'}")
    print(f"Average Extraction: {abs(avg_extraction_percentage):.1f}% {'more' if avg_extraction_percentage > 0 else 'fewer'} entities")

if __name__ == "__main__":
    asyncio.run(main())
