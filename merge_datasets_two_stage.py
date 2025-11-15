"""
Two-Stage Dataset Processing for LLM Pretraining

Stage 1: Download and sample datasets by weight → Save raw data locally
Stage 2: Load raw data → Clean and filter → Save processed data

This approach allows you to:
- Download once, clean multiple times with different parameters
- Inspect raw data before cleaning
- Iterate on cleaning logic without re-downloading
- Save bandwidth and time
"""

import os
import json
import gzip
import yaml
import argparse
import random
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoStageDatasetProcessor:
    """Two-stage dataset processing: Download → Clean"""

    def __init__(self, config_path: str = 'smollm3_weighted_config.yaml'):
        """Initialize with configuration"""
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.datasets = self.config.get('datasets', [])
        self.filtering = self.config.get('filtering', {})

    def stage1_download_and_sample(
        self,
        output_folder: str = './raw_data',
        target_total_samples: int = 1000000,
        seed: int = 42
    ):
        """
        Stage 1: Download datasets and sample by weight

        Args:
            output_folder: Where to save raw data
            target_total_samples: Total number of samples to collect across all datasets
            seed: Random seed for reproducibility
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: Download and Sample by Weight")
        logger.info("="*80)
        logger.info(f"Target total samples: {target_total_samples:,}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Random seed: {seed}")

        random.seed(seed)
        os.makedirs(output_folder, exist_ok=True)

        # Calculate samples per dataset based on weight
        total_weight = sum(d.get('weight', 0) for d in self.datasets)

        dataset_samples = {}
        for dataset_info in self.datasets:
            weight = dataset_info.get('weight', 0)
            if weight > 0:
                # Calculate target samples for this dataset
                target = int(target_total_samples * (weight / total_weight))
                dataset_samples[dataset_info['name']] = {
                    'target': target,
                    'weight': weight,
                    'info': dataset_info
                }

        logger.info(f"\nDataset sampling plan:")
        for name, info in sorted(dataset_samples.items(), key=lambda x: x[1]['target'], reverse=True):
            logger.info(f"  {name}: {info['target']:,} samples ({info['weight']*100:.2f}%)")

        # Download and sample each dataset
        for idx, (dataset_name, info) in enumerate(dataset_samples.items(), 1):
            logger.info(f"\n[{idx}/{len(dataset_samples)}] Processing: {dataset_name}")

            dataset_info = info['info']
            target_samples = info['target']

            try:
                self._download_and_sample_dataset(
                    dataset_info=dataset_info,
                    target_samples=target_samples,
                    output_folder=output_folder,
                    seed=seed
                )
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue

        logger.info("\n" + "="*80)
        logger.info("Stage 1 Complete!")
        logger.info("="*80)

    def _download_and_sample_dataset(
        self,
        dataset_info: Dict,
        target_samples: int,
        output_folder: str,
        seed: int
    ):
        """Download and randomly sample a single dataset"""
        dataset_name = dataset_info['name']
        config = dataset_info.get('config', None)
        text_field = dataset_info.get('text_field', 'text')
        split = dataset_info.get('split', 'train')

        logger.info(f"  Config: {config if config else 'default'}")
        logger.info(f"  Text field: {text_field}")
        logger.info(f"  Target samples: {target_samples:,}")

        # Create output directory
        safe_name = dataset_name.replace('/', '_')
        if config:
            safe_name += f"_{config.replace('/', '_')}"

        dataset_output_dir = os.path.join(output_folder, safe_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        output_file = os.path.join(dataset_output_dir, 'raw_data.jsonl.gz')

        # Check if already processed
        if os.path.exists(output_file):
            logger.info(f"  ✓ Already exists: {output_file}")
            return

        # Load dataset
        logger.info(f"  Loading dataset from HuggingFace...")
        try:
            if config:
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    streaming=True,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"  Failed to load dataset: {e}")
            return

        # Optimized sampling using HuggingFace's shuffle + take
        # This is MUCH faster than manual reservoir sampling
        logger.info(f"  Sampling {target_samples:,} documents...")

        # Shuffle with large buffer for good randomness, then take exact amount needed
        # Buffer size: larger = better randomness, but more memory
        # 100K buffer is a good balance
        buffer_size = min(100000, target_samples * 10)
        shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        sampled_dataset = shuffled_dataset.take(target_samples * 2)  # Take 2x to account for filtered samples

        collected = 0
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for example in tqdm(sampled_dataset, desc=f"  Sampling {dataset_name}", total=target_samples):
                # Extract text
                text = self._extract_text(example, text_field)
                if not text:
                    continue

                # Save document
                doc = {
                    'text': text,
                    'metadata': {
                        'source': dataset_name,
                        'config': config or 'default',
                        'weight': dataset_info.get('weight', 0),
                        'original_index': collected
                    }
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                collected += 1

                # Stop when we have enough
                if collected >= target_samples:
                    break

        logger.info(f"  ✓ Saved: {output_file}")
        logger.info(f"  Collected: {collected:,} samples")

    def _extract_text(self, example: Dict, text_field: str) -> Optional[str]:
        """Extract text from example using field name"""
        try:
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

    def stage2_clean_and_filter(
        self,
        input_folder: str = './raw_data',
        output_folder: str = './cleaned_data'
    ):
        """
        Stage 2: Load raw data, clean and filter

        Args:
            input_folder: Where raw data is stored
            output_folder: Where to save cleaned data
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: Clean and Filter")
        logger.info("="*80)
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")

        os.makedirs(output_folder, exist_ok=True)

        # Import cleaning module
        from clean_and_filter import DatasetCleaner
        cleaner = DatasetCleaner(self.config)

        # Find all raw data files
        raw_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.jsonl.gz') or file.endswith('.jsonl'):
                    raw_files.append(os.path.join(root, file))

        logger.info(f"\nFound {len(raw_files)} raw data files")

        # Process each file
        total_processed = 0
        total_kept = 0

        for idx, raw_file in enumerate(raw_files, 1):
            logger.info(f"\n[{idx}/{len(raw_files)}] Processing: {os.path.basename(raw_file)}")

            # Determine output file
            rel_path = os.path.relpath(raw_file, input_folder)
            output_file = os.path.join(output_folder, rel_path.replace('raw_data', 'cleaned_data'))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Clean and filter
            processed, kept = self._clean_file(raw_file, output_file, cleaner)
            total_processed += processed
            total_kept += kept

            logger.info(f"  Processed: {processed:,}, Kept: {kept:,} ({kept/max(1,processed)*100:.1f}%)")

        logger.info("\n" + "="*80)
        logger.info("Stage 2 Complete!")
        logger.info("="*80)
        logger.info(f"Total processed: {total_processed:,}")
        logger.info(f"Total kept: {total_kept:,} ({total_kept/max(1,total_processed)*100:.1f}%)")

        # Print cleaning statistics
        cleaner.print_stats()

    def _clean_file(self, input_file: str, output_file: str, cleaner) -> tuple:
        """Clean a single file"""

        # Check if already processed
        if os.path.exists(output_file):
            logger.info(f"  ✓ Already cleaned: {output_file}")
            # Count lines
            with gzip.open(output_file, 'rt') if output_file.endswith('.gz') else open(output_file, 'r') as f:
                kept = sum(1 for _ in f)
            return kept, kept

        processed = 0
        kept = 0

        # Open input file
        if input_file.endswith('.gz'):
            input_f = gzip.open(input_file, 'rt', encoding='utf-8')
        else:
            input_f = open(input_file, 'r', encoding='utf-8')

        # Open output file
        if output_file.endswith('.gz'):
            output_f = gzip.open(output_file, 'wt', encoding='utf-8')
        else:
            output_f = open(output_file, 'w', encoding='utf-8')

        try:
            for line in tqdm(input_f, desc=f"  Cleaning"):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    processed += 1

                    # Clean text
                    cleaned_text = cleaner.process_text(doc['text'], min_relevance=0.0)

                    if cleaned_text:
                        # Keep metadata
                        doc['text'] = cleaned_text
                        output_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        kept += 1

                except json.JSONDecodeError:
                    continue

        finally:
            input_f.close()
            output_f.close()

        return processed, kept


def main():
    parser = argparse.ArgumentParser(
        description='Two-stage dataset processing: Download → Clean'
    )

    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        required=True,
        help='Stage to run: 1 (download & sample) or 2 (clean & filter)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='smollm3_weighted_config.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--raw-folder',
        type=str,
        default='./raw_data',
        help='Folder for raw data (Stage 1 output, Stage 2 input)'
    )
    parser.add_argument(
        '--cleaned-folder',
        type=str,
        default='./cleaned_data',
        help='Folder for cleaned data (Stage 2 output)'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=1000000,
        help='Target total samples across all datasets (Stage 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (Stage 1)'
    )

    args = parser.parse_args()

    processor = TwoStageDatasetProcessor(config_path=args.config)

    if args.stage == 1:
        # Stage 1: Download and sample
        processor.stage1_download_and_sample(
            output_folder=args.raw_folder,
            target_total_samples=args.target_samples,
            seed=args.seed
        )

    elif args.stage == 2:
        # Stage 2: Clean and filter
        processor.stage2_clean_and_filter(
            input_folder=args.raw_folder,
            output_folder=args.cleaned_folder
        )

    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()
