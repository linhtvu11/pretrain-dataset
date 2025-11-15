"""
Simple DataTrove Example for Testing
Start here to understand how DataTrove works before running the full pipeline
"""

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.filters import (
    LanguageFilter,
    GopherQualityFilter,
    SamplerFilter,
)
from datatrove.pipeline.writers import JsonlWriter


def simple_example():
    """
    Simple example: Process a small dataset with basic filtering
    """
    print("Running simple DataTrove example...")

    # Define a pipeline
    pipeline = [
        # 1. Read from HuggingFace dataset
        HuggingFaceDatasetReader(
            dataset="Skylion007/openwebtext",  # Small public dataset
            dataset_options={
                "split": "train",
                "streaming": True,
            },
            text_key="text",
            default_metadata={"source": "openwebtext"},
        ),

        # 2. Sample only 0.1% of documents (for quick testing)
        SamplerFilter(rate=0.001, seed=42),

        # 3. Filter by language (English only)
        LanguageFilter(
            languages=["en"],
            language_threshold=0.65,
        ),

        # 4. Quality filter
        GopherQualityFilter(
            min_doc_words=50,  # At least 50 words
            max_doc_words=100000,
            min_avg_word_length=3,
            max_avg_word_length=10,
        ),

        # 5. Write output
        JsonlWriter(
            output_folder="./test_output",
            output_filename="${rank}.jsonl.gz",
            compression="gzip",
        ),
    ]

    # Execute the pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=2,  # 2 parallel tasks
        workers=2,  # 2 workers per task
        logging_dir="./test_logs",
    )

    print("Starting pipeline execution...")
    executor.run()
    print("✓ Complete! Check ./test_output for results")


def vietnamese_example():
    """
    Example processing Vietnamese dataset
    """
    print("Running Vietnamese dataset example...")

    pipeline = [
        # Read Vietnamese news dataset
        HuggingFaceDatasetReader(
            dataset="vietgpt/binhvq_news_vi",
            dataset_options={
                "split": "train",
                "streaming": True,
            },
            text_key="text",
            default_metadata={"source": "vietnamese_news"},
        ),

        # Sample 1% for testing
        SamplerFilter(rate=0.01, seed=42),

        # Filter for Vietnamese language
        LanguageFilter(
            languages=["vi"],
            language_threshold=0.5,
        ),

        # Quality filter
        GopherQualityFilter(
            min_doc_words=30,
            max_doc_words=50000,
        ),

        # Write output
        JsonlWriter(
            output_folder="./test_output_vietnamese",
            output_filename="${rank}.jsonl.gz",
            compression="gzip",
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=2,
        workers=2,
        logging_dir="./test_logs_vietnamese",
    )

    print("Starting Vietnamese pipeline...")
    executor.run()
    print("✓ Complete! Check ./test_output_vietnamese for results")


def multilingual_example():
    """
    Example processing FineWeb-2 with multiple languages
    """
    print("Running multilingual FineWeb-2 example...")

    # Process English subset
    pipeline_en = [
        HuggingFaceDatasetReader(
            dataset="HuggingFaceFW/fineweb-2",
            dataset_options={
                "name": "eng_Latn",  # English in Latin script
                "split": "train",
                "streaming": True,
            },
            text_key="text",
            default_metadata={"source": "fineweb2", "lang": "en"},
        ),

        SamplerFilter(rate=0.0001, seed=42),  # 0.01% for testing

        LanguageFilter(languages=["en"], language_threshold=0.7),

        GopherQualityFilter(min_doc_words=50, max_doc_words=100000),

        JsonlWriter(
            output_folder="./test_output_fineweb/en",
            output_filename="${rank}.jsonl.gz",
            compression="gzip",
        ),
    ]

    # Execute English pipeline
    print("Processing English subset...")
    executor_en = LocalPipelineExecutor(
        pipeline=pipeline_en,
        tasks=1,
        workers=2,
        logging_dir="./test_logs_fineweb/en",
    )
    executor_en.run()

    print("✓ Complete! Check ./test_output_fineweb for results")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "simple":
            simple_example()
        elif example == "vietnamese":
            vietnamese_example()
        elif example == "multilingual":
            multilingual_example()
        else:
            print(f"Unknown example: {example}")
            print("Available examples: simple, vietnamese, multilingual")
    else:
        print("DataTrove Simple Examples")
        print("=" * 60)
        print("\nAvailable examples:")
        print("  python datatrove_simple_example.py simple       - Basic English dataset")
        print("  python datatrove_simple_example.py vietnamese   - Vietnamese dataset")
        print("  python datatrove_simple_example.py multilingual - FineWeb-2 multilingual")
        print("\nRun one of the examples to get started!")
