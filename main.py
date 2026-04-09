import torch
import argparse
import logging
from pathlib import Path

from transformer.training.trainer import Trainer
from transformer.model.transformer import create_model
from transformer.data_.dataset import create_dataloaders
from transformer.training.metrics import MetricsTracker


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Pose-Only Sign Language Model")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    if args.config:
        from transformer.core.config import Config
        config = Config.from_yaml(args.config)
    else:
        from transformer.core.config import get_pose_only_config
        config = get_pose_only_config()
    
    # Override from args
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Validate
    config.validate()
    
    logger.info(f"Training")
    logger.info(f"Feature type: {config.model.feature_type.value}")
    logger.info(f"Pose feature dim: {config.data.pose_feature_dim}")
    logger.info(f"Num classes: {config.data.num_classes}")
    logger.info(f"Max seq length: {config.data.max_seq_length}")
    logger.info(f"Hidden dim: {config.model.hidden_dim}")
    logger.info(f"Num layers: {config.model.num_layers}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Cross-modal attention: {config.model.use_cross_modal_attention}")
    
    # Save config
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, seed=args.seed
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model, config.data, device)
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        resume_from=Path(args.resume) if args.resume else None
    )
    
    history = trainer.train()
    
    # Save final model with config
    final_path = output_dir
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "history": {
            "best_val_accuracy": history.best_val_accuracy,
            "best_epoch": history.best_epoch,
        },
        "feature_type": config.model.feature_type.value,
    }, final_path)
    logger.info(f"Final model saved to {final_path}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set")
    model.eval()
    
    test_metrics = MetricsTracker(config.data.num_classes)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(
                pose_features=batch.pose_features,
                attention_mask=batch.attention_mask
            )
            loss = torch.nn.functional.cross_entropy(output.logits, batch.labels)
            test_metrics.update(output.logits, batch.labels, loss.item())
    
    test_results = test_metrics.compute(compute_per_class=True)
    logger.info(
        f"\nTest Results:\n"
        f"  Accuracy: {test_results.accuracy:.4f}\n"
        f"  Top-5: {test_results.top5_accuracy:.4f}\n"
        f"  Top-10: {test_results.top10_accuracy:.4f}\n"
        f"  Loss: {test_results.loss:.4f}"
    )

if __name__ == "__main__":
    main()
