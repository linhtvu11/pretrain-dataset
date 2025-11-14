"""
Dataset Merging Script for LLM Pretraining
Loads multiple HuggingFace datasets, cleans and filters them, and merges into one dataset.
"""

import os
import yaml
import argparse
from typing import Dict, List, Optional, Iterator
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, interleave_datasets
from tqdm import tqdm
import logging
import numpy as np
from clean_and_filter import DatasetCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetMerger:
    """Handles loading, cleaning, and merging of multiple datasets"""

    def __init__(self, config_path: str = 'datasets_config.yaml', use_weights: bool = False):
        """Initialize with configuration file

        Args:
            config_path: Path to YAML configuration file
            use_weights: Whether to use dataset weights for sampling
        """
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.cleaner = DatasetCleaner(self.config)
        self.merged_data = []
        self.use_weights = use_weights

        if self.use_weights:
            logger.info("Weighted sampling enabled")

    def load_single_dataset(
        self,
        dataset_info: Dict,
        language: str = 'english',
        max_samples: Optional[int] = None
    ) -> Iterator[str]:
        """
        Load a single dataset and yield cleaned texts

        Args:
            dataset_info: Dictionary containing dataset configuration
            language: Language of the dataset ('english' or 'vietnamese')
            max_samples: Maximum number of samples to process (None for all)

        Yields:
            Cleaned text strings
        """
        dataset_name = dataset_info['name']
        text_field = dataset_info['text_field']
        split = dataset_info.get('split', 'train')
        streaming = dataset_info.get('streaming', True)
        config = dataset_info.get('config', None)  # Optional config/subset name

        logger.info(f"Loading dataset: {dataset_name} ({language})" +
                   (f" [config: {config}]" if config else ""))

        try:
            # Load dataset
            dataset = load_dataset(
                dataset_name,
                config if config else None,  # Pass config as second argument
                split=split,
                streaming=streaming,
                trust_remote_code=True
            )

            # Process texts
            count = 0
            kept_count = 0

            if streaming:
                # Streaming mode
                iterator = iter(dataset)
                for example in tqdm(iterator, desc=f"Processing {dataset_name}"):
                    if max_samples and count >= max_samples:
                        break

                    count += 1

                    # Extract text
                    text = self._extract_text(example, text_field)
                    if not text:
                        continue

                    # Clean and filter
                    cleaned = self.cleaner.process_text(text, min_relevance=0.0)

                    if cleaned:
                        kept_count += 1
                        yield cleaned

            else:
                # Non-streaming mode
                total = min(len(dataset), max_samples) if max_samples else len(dataset)

                for i in tqdm(range(total), desc=f"Processing {dataset_name}"):
                    example = dataset[i]
                    count += 1

                    # Extract text
                    text = self._extract_text(example, text_field)
                    if not text:
                        continue

                    # Clean and filter
                    cleaned = self.cleaner.process_text(text, min_relevance=0.0)

                    if cleaned:
                        kept_count += 1
                        yield cleaned

            logger.info(
                f"Completed {dataset_name}: "
                f"Processed {count}, Kept {kept_count} "
                f"({kept_count/max(1, count)*100:.2f}%)"
            )

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")

    def _extract_text(self, example: Dict, text_field: str) -> Optional[str]:
        """Extract text from example using field name"""
        try:
            # Handle nested fields (e.g., "data.text")
            fields = text_field.split('.')
            value = example
            for field in fields:
                value = value[field]

            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                return ' '.join(str(v) for v in value)
            else:
                return str(value)

        except (KeyError, TypeError, AttributeError):
            return None

    def merge_all_datasets_weighted(
        self,
        max_samples: Optional[int] = None,
        seed: int = 42
    ) -> List[str]:
        """
        Merge all datasets using weighted sampling

        Args:
            max_samples: Maximum total samples to generate (None for unlimited)
            seed: Random seed for reproducibility

        Returns:
            List of cleaned texts
        """
        logger.info("Loading datasets for weighted sampling...")

        all_datasets = []
        all_weights = []
        all_names = []

        # Collect all datasets (both English and Vietnamese are in same config)
        for dataset_info in self.config.get('datasets', []):
            weight = dataset_info.get('weight')
            if weight is None or weight == 0:
                logger.warning(f"Skipping {dataset_info['name']}: no weight defined")
                continue

            dataset_name = dataset_info['name']
            text_field = dataset_info['text_field']
            config = dataset_info.get('config', None)
            split = dataset_info.get('split', 'train')

            try:
                logger.info(f"Loading {dataset_name}" +
                           (f" [config: {config}]" if config else "") +
                           f" [weight: {weight}]")

                # Load dataset in streaming mode
                dataset = load_dataset(
                    dataset_name,
                    config if config else None,
                    split=split,
                    streaming=True,
                    trust_remote_code=True
                )

                all_datasets.append(dataset)
                all_weights.append(weight)
                all_names.append(dataset_name + (f"/{config}" if config else ""))

            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {e}")
                continue

        if not all_datasets:
            logger.error("No datasets loaded!")
            return []

        # Normalize weights
        total_weight = sum(all_weights)
        probabilities = [w / total_weight for w in all_weights]

        logger.info(f"\nDataset weights (normalized):")
        for name, prob in zip(all_names, probabilities):
            logger.info(f"  {name}: {prob:.4f} ({prob*100:.2f}%)")

        # Interleave datasets according to weights
        logger.info("\nInterleaving datasets with weighted sampling...")
        interleaved = interleave_datasets(
            all_datasets,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="all_exhausted"
        )

        # Process interleaved dataset
        merged_texts = []
        count = 0
        kept_count = 0

        for example in tqdm(interleaved, desc="Processing weighted samples", total=max_samples):
            if max_samples and count >= max_samples:
                break

            count += 1

            # Try to extract text (try common field names)
            text = None
            for dataset_info in self.config.get('datasets', []):
                text_field = dataset_info['text_field']
                text = self._extract_text(example, text_field)
                if text:
                    break

            if not text:
                continue

            # Clean and filter
            cleaned = self.cleaner.process_text(text, min_relevance=0.0)

            if cleaned:
                kept_count += 1
                merged_texts.append(cleaned)

        logger.info(
            f"\nWeighted sampling completed: "
            f"Processed {count}, Kept {kept_count} "
            f"({kept_count/max(1, count)*100:.2f}%)"
        )

        return merged_texts

    def merge_all_datasets(
        self,
        include_english: bool = True,
        include_vietnamese: bool = True,
        max_samples_per_dataset: Optional[int] = None
    ) -> List[str]:
        """
        Merge all datasets from configuration (simple concatenation without weights)

        Args:
            include_english: Whether to include English datasets
            include_vietnamese: Whether to include Vietnamese datasets
            max_samples_per_dataset: Max samples per dataset (None for all)

        Returns:
            List of cleaned texts
        """
        merged_texts = []

        # Process English datasets
        if include_english:
            logger.info("Processing English datasets...")
            for dataset_info in self.config.get('english_datasets', []):
                for text in self.load_single_dataset(
                    dataset_info,
                    language='english',
                    max_samples=max_samples_per_dataset
                ):
                    merged_texts.append(text)

        # Process Vietnamese datasets
        if include_vietnamese:
            logger.info("Processing Vietnamese datasets...")
            for dataset_info in self.config.get('vietnamese_datasets', []):
                for text in self.load_single_dataset(
                    dataset_info,
                    language='vietnamese',
                    max_samples=max_samples_per_dataset
                ):
                    merged_texts.append(text)

        logger.info(f"Total merged texts: {len(merged_texts)}")
        return merged_texts

    def create_huggingface_dataset(
        self,
        texts: List[str],
        train_test_split: float = 0.95
    ) -> DatasetDict:
        """
        Create a HuggingFace Dataset from list of texts

        Args:
            texts: List of text strings
            train_test_split: Ratio for train/test split

        Returns:
            DatasetDict with train and test splits
        """
        logger.info("Creating HuggingFace dataset...")

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        # Split into train/test
        split_idx = int(len(texts) * train_test_split)
        train_dataset = Dataset.from_dict({"text": texts[:split_idx]})
        test_dataset = Dataset.from_dict({"text": texts[split_idx:]})

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        logger.info(f"Created dataset with {len(train_dataset)} train and {len(test_dataset)} test samples")

        return dataset_dict

    def push_to_hub(
        self,
        dataset: DatasetDict,
        repo_name: Optional[str] = None,
        private: bool = False
    ):
        """
        Push dataset to HuggingFace Hub

        Args:
            dataset: DatasetDict to push
            repo_name: Repository name (defaults to config)
            private: Whether to make the dataset private
        """
        if repo_name is None:
            repo_name = self.config.get('output', {}).get('hub_repo')

        if not repo_name:
            logger.error("No repository name provided and none found in config")
            return

        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_name}")

        try:
            dataset.push_to_hub(
                repo_name,
                private=private
            )
            logger.info(f"Successfully pushed to {repo_name}")

        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
            logger.info("Make sure you're logged in with: huggingface-cli login")

    def save_local(self, dataset: DatasetDict, output_dir: str = "./merged_dataset"):
        """Save dataset locally"""
        logger.info(f"Saving dataset locally to {output_dir}")
        dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Merge and clean HuggingFace datasets for LLM pretraining'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='datasets_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--english-only',
        action='store_true',
        help='Only process English datasets'
    )
    parser.add_argument(
        '--vietnamese-only',
        action='store_true',
        help='Only process Vietnamese datasets'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per dataset (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./merged_dataset',
        help='Local output directory'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push to HuggingFace Hub'
    )
    parser.add_argument(
        '--hub-repo',
        type=str,
        default=None,
        help='HuggingFace Hub repository name'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the dataset private on HuggingFace Hub'
    )
    parser.add_argument(
        '--no-save-local',
        action='store_true',
        help='Skip saving locally'
    )
    parser.add_argument(
        '--use-weights',
        action='store_true',
        help='Use dataset weights for sampling (requires weighted config file)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for weighted sampling (default: 42)'
    )

    args = parser.parse_args()

    # Initialize merger
    merger = DatasetMerger(args.config, use_weights=args.use_weights)

    # Merge datasets
    if args.use_weights:
        # Use weighted sampling
        logger.info("Using weighted sampling mode")
        merged_texts = merger.merge_all_datasets_weighted(
            max_samples=args.max_samples,
            seed=args.seed
        )
    else:
        # Use simple concatenation
        logger.info("Using simple concatenation mode")
        # Determine which datasets to include
        include_english = not args.vietnamese_only
        include_vietnamese = not args.english_only

        merged_texts = merger.merge_all_datasets(
            include_english=include_english,
            include_vietnamese=include_vietnamese,
            max_samples_per_dataset=args.max_samples
        )

    if not merged_texts:
        logger.error("No texts were merged! Check your configuration and dataset access.")
        return

    # Create HuggingFace dataset
    dataset = merger.create_huggingface_dataset(merged_texts)

    # Save locally
    if not args.no_save_local:
        merger.save_local(dataset, args.output_dir)

    # Push to hub
    if args.push_to_hub:
        merger.push_to_hub(
            dataset,
            repo_name=args.hub_repo,
            private=args.private
        )

    # Print final statistics
    merger.cleaner.print_stats()

    logger.info("Process completed!")


if __name__ == "__main__":
    main()
