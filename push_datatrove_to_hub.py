"""
Push DataTrove Processed Data to HuggingFace Hub

This script reads the JSONL.gz files produced by smollm3_stage1_datatrove_pipeline.py
and pushes them to HuggingFace Hub as a dataset.
"""

import os
import json
import gzip
import argparse
from pathlib import Path
from typing import Iterator, Dict, List
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_jsonl_gz(file_path: str) -> Iterator[Dict]:
    """
    Read a gzipped JSONL file and yield documents

    Args:
        file_path: Path to .jsonl.gz file

    Yields:
        Document dictionaries
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {file_path}: {e}")
                    continue


def find_all_jsonl_files(input_folder: str) -> List[str]:
    """
    Recursively find all .jsonl.gz files in a folder

    Args:
        input_folder: Root folder to search

    Returns:
        List of file paths
    """
    jsonl_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jsonl.gz') or file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))

    return sorted(jsonl_files)


def load_documents_from_folder(
    input_folder: str,
    max_documents: int = None,
    sample_ratio: float = 1.0
) -> List[Dict]:
    """
    Load all documents from DataTrove output folder

    Args:
        input_folder: Folder containing processed JSONL files
        max_documents: Maximum number of documents to load (None for all)
        sample_ratio: Ratio of documents to sample (0.0-1.0)

    Returns:
        List of document dictionaries
    """
    logger.info(f"Searching for JSONL files in {input_folder}")

    # Find all JSONL files
    jsonl_files = find_all_jsonl_files(input_folder)

    if not jsonl_files:
        logger.error(f"No JSONL files found in {input_folder}")
        return []

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # Load documents
    documents = []
    total_files = len(jsonl_files)

    for idx, file_path in enumerate(jsonl_files, 1):
        logger.info(f"Loading file {idx}/{total_files}: {file_path}")

        file_docs = 0
        for doc in read_jsonl_gz(file_path) if file_path.endswith('.gz') else read_jsonl(file_path):
            # Sample if needed
            if sample_ratio < 1.0:
                import random
                if random.random() > sample_ratio:
                    continue

            documents.append(doc)
            file_docs += 1

            # Check max documents limit
            if max_documents and len(documents) >= max_documents:
                logger.info(f"Reached max documents limit: {max_documents}")
                break

        logger.info(f"  Loaded {file_docs} documents from {os.path.basename(file_path)}")

        if max_documents and len(documents) >= max_documents:
            break

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def read_jsonl(file_path: str) -> Iterator[Dict]:
    """Read uncompressed JSONL file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {file_path}: {e}")
                    continue


def create_dataset_from_documents(
    documents: List[Dict],
    train_test_split: float = 0.95
) -> DatasetDict:
    """
    Create HuggingFace Dataset from documents

    Args:
        documents: List of document dictionaries
        train_test_split: Ratio for train/test split

    Returns:
        DatasetDict with train and test splits
    """
    logger.info("Creating HuggingFace Dataset...")

    # Extract fields
    texts = []
    ids = []
    sources = []
    weights = []
    configs = []

    for doc in tqdm(documents, desc="Processing documents"):
        texts.append(doc.get('text', ''))
        ids.append(doc.get('id', ''))

        metadata = doc.get('metadata', {})
        sources.append(metadata.get('source', 'unknown'))
        weights.append(metadata.get('weight', 0.0))
        configs.append(metadata.get('config', 'default'))

    # Create dataset
    data_dict = {
        'text': texts,
        'id': ids,
        'source': sources,
        'weight': weights,
        'config': configs,
    }

    full_dataset = Dataset.from_dict(data_dict)

    # Split into train/test
    logger.info(f"Splitting dataset: {train_test_split*100:.1f}% train, {(1-train_test_split)*100:.1f}% test")

    split_idx = int(len(documents) * train_test_split)

    train_data = {k: v[:split_idx] for k, v in data_dict.items()}
    test_data = {k: v[split_idx:] for k, v in data_dict.items()}

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
    })

    logger.info(f"Created dataset with {len(train_dataset)} train and {len(test_dataset)} test samples")

    return dataset_dict


def push_to_hub(
    dataset: DatasetDict,
    repo_name: str,
    private: bool = False,
    token: str = None
):
    """
    Push dataset to HuggingFace Hub

    Args:
        dataset: DatasetDict to push
        repo_name: Repository name (username/dataset-name)
        private: Whether to make the dataset private
        token: HuggingFace API token (optional, uses cached token if not provided)
    """
    logger.info(f"Pushing dataset to HuggingFace Hub: {repo_name}")
    logger.info(f"  Private: {private}")

    try:
        dataset.push_to_hub(
            repo_name,
            private=private,
            token=token,
        )
        logger.info(f"✓ Successfully pushed to https://huggingface.co/datasets/{repo_name}")

    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Make sure you're logged in: huggingface-cli login")
        logger.info("  2. Check your repository name format: username/dataset-name")
        logger.info("  3. Verify you have write permissions to the repository")
        raise


def save_local(
    dataset: DatasetDict,
    output_dir: str
):
    """
    Save dataset locally

    Args:
        dataset: DatasetDict to save
        output_dir: Directory to save to
    """
    logger.info(f"Saving dataset locally to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    logger.info(f"✓ Dataset saved to {output_dir}")

    # Print dataset info
    logger.info("\nDataset info:")
    logger.info(f"  Train samples: {len(dataset['train'])}")
    logger.info(f"  Test samples: {len(dataset['test'])}")
    logger.info(f"  Features: {dataset['train'].features}")


def print_dataset_statistics(documents: List[Dict]):
    """Print statistics about the loaded documents"""
    logger.info("\n" + "="*80)
    logger.info("Dataset Statistics")
    logger.info("="*80)

    # Count by source
    source_counts = {}
    for doc in documents:
        source = doc.get('metadata', {}).get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info(f"\nTotal documents: {len(documents)}")
    logger.info(f"\nDocuments by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(documents) * 100
        logger.info(f"  {source}: {count:,} ({pct:.2f}%)")

    # Text length statistics
    text_lengths = [len(doc.get('text', '')) for doc in documents[:10000]]  # Sample for speed
    if text_lengths:
        import numpy as np
        logger.info(f"\nText length statistics (sampled from first 10k docs):")
        logger.info(f"  Min: {min(text_lengths):,} chars")
        logger.info(f"  Max: {max(text_lengths):,} chars")
        logger.info(f"  Mean: {np.mean(text_lengths):,.0f} chars")
        logger.info(f"  Median: {np.median(text_lengths):,.0f} chars")

    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Push DataTrove processed data to HuggingFace Hub'
    )

    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Folder containing DataTrove output (JSONL.gz files)'
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        required=True,
        help='HuggingFace repository name (username/dataset-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the dataset private'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Local directory to save dataset (optional)'
    )
    parser.add_argument(
        '--max-documents',
        type=int,
        default=None,
        help='Maximum number of documents to load (for testing)'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=1.0,
        help='Ratio of documents to sample (0.0-1.0, default: 1.0 = all)'
    )
    parser.add_argument(
        '--train-test-split',
        type=float,
        default=0.95,
        help='Train/test split ratio (default: 0.95)'
    )
    parser.add_argument(
        '--no-push',
        action='store_true',
        help='Skip pushing to HuggingFace Hub (only save locally)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (optional)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return

    if args.sample_ratio < 0 or args.sample_ratio > 1:
        logger.error("sample-ratio must be between 0 and 1")
        return

    # Load documents
    logger.info("\n" + "="*80)
    logger.info("Loading DataTrove Processed Data")
    logger.info("="*80 + "\n")

    documents = load_documents_from_folder(
        input_folder=args.input_folder,
        max_documents=args.max_documents,
        sample_ratio=args.sample_ratio,
    )

    if not documents:
        logger.error("No documents loaded! Check your input folder.")
        return

    # Print statistics
    print_dataset_statistics(documents)

    # Create HuggingFace dataset
    dataset = create_dataset_from_documents(
        documents=documents,
        train_test_split=args.train_test_split,
    )

    # Save locally if requested
    if args.output_dir:
        save_local(dataset, args.output_dir)

    # Push to HuggingFace Hub
    if not args.no_push:
        push_to_hub(
            dataset=dataset,
            repo_name=args.repo_name,
            private=args.private,
            token=args.token,
        )
    else:
        logger.info("Skipping push to HuggingFace Hub (--no-push flag set)")

    logger.info("\n✓ All done!")
    logger.info(f"\nDataset summary:")
    logger.info(f"  Total documents: {len(documents):,}")
    logger.info(f"  Train samples: {len(dataset['train']):,}")
    logger.info(f"  Test samples: {len(dataset['test']):,}")

    if not args.no_push:
        logger.info(f"\n🎉 Dataset available at: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()
