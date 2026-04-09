import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass, field, asdict
import random
import numpy as np
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:

    id: str
    label: str
    video_path: str
    split: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetStats:

    total_samples: int
    num_classes: int
    samples_per_class: Dict[str, int]
    split_distribution: Dict[str, int] = field(default_factory=dict)
    class_imbalance_ratio: float = 1.0
    min_samples_per_class: int = 0
    max_samples_per_class: int = 0
    mean_samples_per_class: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        lines = [
            "DATASET STATISTICS",
            f"Total samples:     {self.total_samples}",
            f"Number of classes: {self.num_classes}",
            f"Min samples/class: {self.min_samples_per_class}",
            f"Max samples/class: {self.max_samples_per_class}",
            f"Mean samples/class: {self.mean_samples_per_class:.2f}",
            f"Imbalance ratio:   {self.class_imbalance_ratio:.2f}",
        ]
        
        if self.split_distribution:
            lines.append("-" * 50)
            lines.append("Split distribution:")
            for split, count in self.split_distribution.items():
                pct = (count / self.total_samples) * 100
                lines.append(f"  {split}: {count} ({pct:.1f}%)")
                
        lines.append("=" * 50)
        return "\n".join(lines)


class DatasetPreparator:
    
    video_extensions = {'.mp4'}
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        seed: int = 42
    ):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize random state
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(
            f"DatasetPreparator initialized: "
            f"data_dir={self.data_dir}, seed={seed}"
        )
        
    def discover_dataset(self) -> Tuple[List[SampleInfo], DatasetStats]:

        samples = []
        class_counts = Counter()
        
        # Iterate through class folders
        class_folders = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not class_folders:
            raise ValueError(
                f"No class folders found in {self.data_dir}. "
                "Expected structure: dataset/class_name/videos.mp4"
            )
            
        logger.info(f"Found {len(class_folders)} class folders")
        
        for class_folder in class_folders:
            class_name = class_folder.name
            
            # Find videos in this class folder
            videos = self._find_videos(class_folder)
            
            for video_path in videos:
                # Create unique sample ID: classname_videoname
                sample_id = f"{class_name}_{video_path.stem}"
                
                sample = SampleInfo(
                    id=sample_id,
                    label=class_name,
                    video_path=str(video_path)
                )
                samples.append(sample)
                class_counts[class_name] += 1
                
        if not samples:
            raise ValueError(
                f"No video samples found in {self.data_dir}. "
                f"Supported formats: {self.video_extensions}"
            )
            
        # Calculate statistics
        counts = list(class_counts.values())
        stats = DatasetStats(
            total_samples=len(samples),
            num_classes=len(class_counts),
            samples_per_class=dict(class_counts),
            min_samples_per_class=min(counts),
            max_samples_per_class=max(counts),
            mean_samples_per_class=np.mean(counts),
            class_imbalance_ratio=max(counts) / min(counts) if min(counts) > 0 else float('inf')
        )
        
        logger.info(f"Discovered {stats.total_samples} samples in {stats.num_classes} classes")
        
        return samples, stats
        
    def _find_videos(self, folder: Path) -> List[Path]:

        seen_resolved = set()
        videos = []
        
        for ext in self.video_extensions:
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for v in folder.glob(pattern):
                    resolved = v.resolve()
                    if resolved not in seen_resolved:
                        seen_resolved.add(resolved)
                        videos.append(v)
        
        return sorted(videos)
        
    def create_splits(
        self,
        samples: List[SampleInfo],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[SampleInfo]]:
            
        # Group samples by class
        class_samples = {}
        for sample in samples:
            if sample.label not in class_samples:
                class_samples[sample.label] = []
            class_samples[sample.label].append(sample)
            
        splits = {"train": [], "val": [], "test": []}
        
        # Stratified split for each class
        for class_name, class_data in class_samples.items():
            random.shuffle(class_data)
            
            n = len(class_data)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            
            # Ensure at least one sample in each split if possible
            if n >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = n - n_train - n_val
                
                if n_test < 1:
                    n_train -= 1
                    n_test = 1
            else:
                # Very few samples: prioritize train
                n_train = n
                n_val = 0
                n_test = 0
                
            train_samples = class_data[:n_train]
            val_samples = class_data[n_train:n_train + n_val]
            test_samples = class_data[n_train + n_val:]
            
            for sample in train_samples:
                sample.split = "train"
            for sample in val_samples:
                sample.split = "val"
            for sample in test_samples:
                sample.split = "test"
                
            splits["train"].extend(train_samples)
            splits["val"].extend(val_samples)
            splits["test"].extend(test_samples)
            
        logger.info(
            f"Created splits: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )
        
        return splits
        
    def save_metadata(
        self,
        samples: List[SampleInfo],
        stats: DatasetStats
    ) -> Dict[str, Path]:

        output_files = {}
        
        # Convert samples to dicts
        samples_dicts = [s.to_dict() for s in samples]
        
        # Save all samples
        all_samples_path = self.output_dir / "all_samples.json"
        with open(all_samples_path, 'w', encoding='utf-8') as f:
            json.dump(samples_dicts, f, indent=2, ensure_ascii=False)
        output_files["all_samples"] = all_samples_path
        
        # Save split-specific files
        for split in ["train", "val", "test"]:
            split_samples = [s for s in samples_dicts if s["split"] == split]
            split_path = self.output_dir / f"{split}_samples.json"
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_samples, f, indent=2, ensure_ascii=False)
            output_files[split] = split_path
            
        # Save label mapping
        unique_labels = sorted(set(s.label for s in samples))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        labels_path = self.output_dir / "labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump({
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "num_classes": len(unique_labels)
            }, f, indent=2, ensure_ascii=False)
        output_files["labels"] = labels_path
        
        # Update stats with split distribution
        stats.split_distribution = {
            "train": len([s for s in samples if s.split == "train"]),
            "val": len([s for s in samples if s.split == "val"]),
            "test": len([s for s in samples if s.split == "test"])
        }
        
        # Save statistics
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            stats_dict = stats.to_dict()
            stats_dict["created_at"] = datetime.now().isoformat()
            stats_dict["seed"] = self.seed
            json.dump(stats_dict, f, indent=2)
        output_files["stats"] = stats_path
        
        logger.info(f"Saved metadata files to {self.output_dir}")
        
        return output_files
        
    def prepare(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[SampleInfo], DatasetStats]:

        logger.info("Starting dataset preparation")
        
        # Discover dataset
        samples, stats = self.discover_dataset()
        
        print(stats)
        
        self.create_splits(samples, train_ratio, val_ratio, test_ratio)
        
        self.save_metadata(samples, stats)
        
        logger.info("Dataset preparation complete")
        
        return samples, stats
        
    def get_video_paths_for_extraction(
        self,
        samples: Optional[List[SampleInfo]] = None
    ) -> Tuple[List[Path], List[str]]:

        if samples is None:
            samples, _ = self.discover_dataset()
            
        video_paths = [Path(s.video_path) for s in samples]
        sample_ids = [s.id for s in samples]
        
        return video_paths, sample_ids


def prepare_dataset_cli(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:

    preparator = DatasetPreparator(data_dir, output_dir, seed)
    samples, stats = preparator.prepare(train_ratio, val_ratio, test_ratio)
    
    print(f"\nDataset prepared successfully")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare sign language dataset from folder structure"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing class folders"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for metadata files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    prepare_dataset_cli(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
