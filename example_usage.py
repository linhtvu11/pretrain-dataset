"""
Example usage of the dataset merger
This script demonstrates how to use the DatasetMerger programmatically
"""

from merge_datasets import DatasetMerger
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def example_basic_usage():
    """Basic usage example"""
    print("\n=== Example 1: Basic Usage ===\n")

    # Initialize merger
    merger = DatasetMerger('datasets_config.yaml')

    # Merge with limited samples (for testing)
    merged_texts = merger.merge_all_datasets(
        include_english=True,
        include_vietnamese=True,
        max_samples_per_dataset=10  # Only 10 samples per dataset for demo
    )

    print(f"\nMerged {len(merged_texts)} texts")

    # Show some examples
    if merged_texts:
        print("\n--- Sample texts ---")
        for i, text in enumerate(merged_texts[:3], 1):
            print(f"\nSample {i}:")
            print(text[:200] + "..." if len(text) > 200 else text)

    # Print statistics
    merger.cleaner.print_stats()


def example_english_only():
    """Process only English datasets"""
    print("\n=== Example 2: English Only ===\n")

    merger = DatasetMerger('datasets_config.yaml')

    merged_texts = merger.merge_all_datasets(
        include_english=True,
        include_vietnamese=False,
        max_samples_per_dataset=5
    )

    print(f"\nMerged {len(merged_texts)} English texts")
    merger.cleaner.print_stats()


def example_create_and_save():
    """Create dataset and save locally"""
    print("\n=== Example 3: Create and Save Dataset ===\n")

    merger = DatasetMerger('datasets_config.yaml')

    # Merge limited samples
    merged_texts = merger.merge_all_datasets(
        include_english=True,
        include_vietnamese=False,
        max_samples_per_dataset=10
    )

    if merged_texts:
        # Create HuggingFace dataset
        dataset = merger.create_huggingface_dataset(merged_texts)

        # Save locally
        merger.save_local(dataset, "./example_output")

        print("\nDataset saved to ./example_output")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")


def example_custom_processing():
    """Example of custom processing"""
    print("\n=== Example 4: Custom Processing ===\n")

    import yaml
    from clean_and_filter import DatasetCleaner

    # Load and modify config
    with open('datasets_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Customize filtering
    config['filtering']['min_length'] = 200  # Longer texts only
    config['filtering']['keep_keywords'].extend(['AI', 'neural network', 'deep learning'])

    # Create cleaner with custom config
    cleaner = DatasetCleaner(config)

    # Test texts
    test_texts = [
        "This is about neural networks and deep learning in AI systems.",
        "Short text",  # Will be filtered
        "A detailed explanation of transformer architectures in natural language processing."
    ]

    print("Processing with custom config (min_length=200):\n")
    for i, text in enumerate(test_texts, 1):
        result = cleaner.process_text(text)
        print(f"Text {i}: {'KEPT' if result else 'FILTERED'}")
        print(f"  Length: {len(text)}")
        print(f"  Preview: {text[:80]}...")
        print()

    cleaner.print_stats()


if __name__ == "__main__":
    # Run examples
    # Uncomment the ones you want to run

    example_basic_usage()

    # example_english_only()

    # example_create_and_save()

    # example_custom_processing()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
