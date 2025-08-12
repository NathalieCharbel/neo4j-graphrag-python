import asyncio
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

load_dotenv()

# Global metrics storage per folder
folder_metrics = {}

async def test_structured_vs_traditional(file_path, folder_name):
    print(f"\nTesting: {file_path.name}")
    
    loader = PdfLoader()
    document = await loader.run(filepath=file_path)
    
    splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = await splitter.run(document.text)
    
    print(f"Chunks: {len(chunks.chunks)}")
    
    # Run each extractor 3 times and calculate averages
    traditional_times = []
    structured_times = []
    traditional_totals = []
    structured_totals = []
    
    for run in range(1):
        print(f"  Run {run + 1}/3")
        
        extractor_traditional = LLMEntityRelationExtractor(structured_output=False)
        start_time = time.time()
        result_traditional = await extractor_traditional.run(chunks)
        traditional_time = time.time() - start_time
        traditional_times.append(traditional_time)
        traditional_totals.append(len(result_traditional.nodes) + len(result_traditional.relationships))
        
        extractor_structured = LLMEntityRelationExtractor(structured_output=True)
        start_time = time.time()
        result_structured = await extractor_structured.run(chunks)
        structured_time = time.time() - start_time
        structured_times.append(structured_time)
        structured_totals.append(len(result_structured.nodes) + len(result_structured.relationships))
    
    # Calculate averages
    avg_traditional_time = sum(traditional_times) / len(traditional_times)
    avg_structured_time = sum(structured_times) / len(structured_times)
    avg_traditional_total = sum(traditional_totals) / len(traditional_totals)
    avg_structured_total = sum(structured_totals) / len(structured_totals)
    
    time_diff = avg_structured_time - avg_traditional_time
    time_percentage = (time_diff / avg_traditional_time) * 100
    
    extraction_diff = avg_structured_total - avg_traditional_total
    extraction_percentage = (extraction_diff / avg_structured_total) * 100
    
    print(f"Traditional: {avg_traditional_time:.2f}s, {avg_traditional_total:.1f} entities")
    print(f"Structured: {avg_structured_time:.2f}s, {avg_structured_total:.1f} entities")
    print(f"Performance: {abs(time_percentage):.1f}% {'slower' if time_diff > 0 else 'faster'}")
    print(f"Extraction: {abs(extraction_percentage):.1f}% {'more' if extraction_diff > 0 else 'fewer'}")
    
    # Store in folder metrics
    if folder_name not in folder_metrics:
        folder_metrics[folder_name] = {
            'traditional_times': [],
            'structured_times': [],
            'traditional_totals': [],
            'structured_totals': []
        }
    
    folder_metrics[folder_name]['traditional_times'].append(avg_traditional_time)
    folder_metrics[folder_name]['structured_times'].append(avg_structured_time)
    folder_metrics[folder_name]['traditional_totals'].append(avg_traditional_total)
    folder_metrics[folder_name]['structured_totals'].append(avg_structured_total)

def load_results():
    results_file = Path("corpus_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

def save_results(folder_name, metrics):
    results_file = Path("corpus_results.json")
    all_results = load_results()
    all_results[folder_name] = {
        'traditional_times': metrics['traditional_times'],
        'structured_times': metrics['structured_times'],
        'traditional_totals': metrics['traditional_totals'],
        'structured_totals': metrics['structured_totals']
    }
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

def plot_results():
    all_results = load_results()
    folders = list(all_results.keys())
    
    # Calculate metrics for each folder
    folder_stats = {}
    for folder in folders:
        metrics = all_results[folder]
        avg_trad_time = np.mean(metrics['traditional_times'])
        avg_struct_time = np.mean(metrics['structured_times'])
        avg_trad_total = np.mean(metrics['traditional_totals'])
        avg_struct_total = np.mean(metrics['structured_totals'])
        
        time_diff = avg_struct_time - avg_trad_time
        time_percentage = (time_diff / avg_trad_time) * 100
        
        extraction_diff = avg_struct_total - avg_trad_total
        extraction_percentage = (extraction_diff / avg_trad_total) * 100
        
        folder_stats[folder] = {
            'avg_trad_time': avg_trad_time,
            'avg_struct_time': avg_struct_time,
            'avg_trad_total': avg_trad_total,
            'avg_struct_total': avg_struct_total,
            'time_percentage': time_percentage,
            'extraction_percentage': extraction_percentage
        }
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time comparison
    trad_times = [folder_stats[f]['avg_trad_time'] for f in folders]
    struct_times = [folder_stats[f]['avg_struct_time'] for f in folders]
    x = np.arange(len(folders))
    width = 0.35
    
    ax1.bar(x - width/2, trad_times, width, label='Traditional', alpha=0.8)
    ax1.bar(x + width/2, struct_times, width, label='Structured', alpha=0.8)
    ax1.set_xlabel('Corpus')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Average Execution Time by Corpus')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folders, rotation=45)
    ax1.legend()
    
    # Extraction comparison
    trad_totals = [folder_stats[f]['avg_trad_total'] for f in folders]
    struct_totals = [folder_stats[f]['avg_struct_total'] for f in folders]
    
    ax2.bar(x - width/2, trad_totals, width, label='Traditional', alpha=0.8)
    ax2.bar(x + width/2, struct_totals, width, label='Structured', alpha=0.8)
    ax2.set_xlabel('Corpus')
    ax2.set_ylabel('Entities/Relations Count')
    ax2.set_title('Average Extraction Count by Corpus')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folders, rotation=45)
    ax2.legend()
    
    # Performance percentage
    time_percentages = [folder_stats[f]['time_percentage'] for f in folders]
    colors = ['red' if p > 0 else 'green' for p in time_percentages]
    
    ax3.bar(folders, time_percentages, color=colors, alpha=0.7)
    ax3.set_xlabel('Corpus')
    ax3.set_ylabel('Performance Difference (%)')
    ax3.set_title('Performance: Structured vs Traditional')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Extraction percentage
    extraction_percentages = [folder_stats[f]['extraction_percentage'] for f in folders]
    colors = ['green' if p > 0 else 'red' for p in extraction_percentages]
    
    ax4.bar(folders, extraction_percentages, color=colors, alpha=0.7)
    ax4.set_xlabel('Corpus')
    ax4.set_ylabel('Extraction Difference (%)')
    ax4.set_title('Extraction: Structured vs Traditional')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.setp(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

async def main():
    # Comment/uncomment one folder at a time
    # folder_name = "medical_papers"
    # folder_name = "threat_intelligence"
    folder_name = "RILA"
    # folder_name = "medical_reports"
    
    corpus_folder = Path(f"examples/data/{folder_name}")
    if not corpus_folder.exists():
        print(f"Folder {corpus_folder} does not exist")
        return
        
    pdf_files = list(corpus_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {corpus_folder}")
        return
    
    print(f"\n=== PROCESSING {folder_name.upper()} ===")
    print(f"Testing {len(pdf_files)} documents")
    
    for pdf_file in pdf_files:
        await test_structured_vs_traditional(pdf_file, folder_name)
    
    # Save results
    save_results(folder_name, folder_metrics[folder_name])
    
    # Print current folder results
    metrics = folder_metrics[folder_name]
    avg_trad_time = np.mean(metrics['traditional_times'])
    avg_struct_time = np.mean(metrics['structured_times'])
    avg_trad_total = np.mean(metrics['traditional_totals'])
    avg_struct_total = np.mean(metrics['structured_totals'])
    
    time_diff = avg_struct_time - avg_trad_time
    time_percentage = (time_diff / avg_trad_time) * 100
    
    extraction_diff = avg_struct_total - avg_trad_total
    extraction_percentage = (extraction_diff / avg_trad_total) * 100
    
    print(f"\n=== {folder_name.upper()} RESULTS ===")
    print(f"Average Traditional Time: {avg_trad_time:.2f}s")
    print(f"Average Structured Time: {avg_struct_time:.2f}s")
    print(f"Average Traditional Extractions: {avg_trad_total:.1f} entities/rels")
    print(f"Average Structured Extractions: {avg_struct_total:.1f} entities/rels")
    print(f"Performance: {abs(time_percentage):.1f}% {'slower' if time_percentage > 0 else 'faster'}")
    print(f"Extraction: {abs(extraction_percentage):.1f}% {'more' if extraction_percentage > 0 else 'fewer'}")

def plot_all_results():
    plot_results()

if __name__ == "__main__":
    asyncio.run(main())
