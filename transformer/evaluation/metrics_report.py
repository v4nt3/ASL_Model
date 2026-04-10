import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader

from transformer.core.config import Config
from transformer.model.transformer import create_model
from transformer.data_.dataset import SignLanguageDataset, collate_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORS = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "bg": "#FAFAFA",
    "text": "#1F2937",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": "white",
    "axes.edgecolor": "#E5E7EB",
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["neutral"],
    "ytick.color": COLORS["neutral"],
    "text.color": COLORS["text"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def plot_training_history(history_path: str, output_dir: Path) -> None:
    if not Path(history_path).exists():
        logger.warning(f"Training history not found at {history_path}, skipping.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    train_metrics = history.get("train", [])
    val_metrics = history.get("val", [])

    if not train_metrics:
        logger.warning("No training metrics in history file.")
        return

    epochs = list(range(1, len(train_metrics) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Historial de Entrenamiento", fontsize=18, fontweight="bold", y=0.98)

    # Loss
    ax = axes[0, 0]
    train_loss = [m.get("loss", 0) for m in train_metrics]
    ax.plot(epochs, train_loss, color=COLORS["primary"], linewidth=2, label="Pérdida Entrenamiento")
    if val_metrics:
        val_loss = [m.get("loss", 0) for m in val_metrics]
        val_epochs = list(range(1, len(val_metrics) + 1))
        ax.plot(val_epochs, val_loss, color=COLORS["danger"], linewidth=2, label="Pérdida Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.set_title("Pérdida")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    train_acc = [m.get("accuracy", 0) for m in train_metrics]
    ax.plot(epochs, train_acc, color=COLORS["primary"], linewidth=2, label="Exactitud Entrenamiento")
    if val_metrics:
        val_acc = [m.get("accuracy", 0) for m in val_metrics]
        ax.plot(val_epochs, val_acc, color=COLORS["secondary"], linewidth=2, label="Exactitud Validación")
        best_epoch = history.get("best_epoch", 0) + 1
        best_acc = history.get("best_val_accuracy", 0)
        ax.axvline(x=best_epoch, color=COLORS["accent"], linestyle="--", alpha=0.7)
        ax.annotate(
            f"Mejor: {best_acc:.4f} (ép. {best_epoch})",
            xy=(best_epoch, best_acc),
            xytext=(best_epoch + 2, best_acc - 0.05),
            arrowprops=dict(arrowstyle="->", color=COLORS["accent"]),
            fontsize=10, color=COLORS["accent"],
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud")
    ax.set_title("Exactitud Top-1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-5 Accuracy
    ax = axes[1, 0]
    train_top5 = [m.get("top5_accuracy", 0) for m in train_metrics]
    ax.plot(epochs, train_top5, color=COLORS["primary"], linewidth=2, label="Entrenamiento Top-5")
    if val_metrics:
        val_top5 = [m.get("top5_accuracy", 0) for m in val_metrics]
        ax.plot(val_epochs, val_top5, color=COLORS["secondary"], linewidth=2, label="Validación Top-5")
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud Top-5")
    ax.set_title("Exactitud Top-5")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-10 Accuracy
    ax = axes[1, 1]
    train_top10 = [m.get("top10_accuracy", 0) for m in train_metrics]
    ax.plot(epochs, train_top10, color=COLORS["primary"], linewidth=2, label="Entrenamiento Top-10")
    if val_metrics:
        val_top10 = [m.get("top10_accuracy", 0) for m in val_metrics]
        ax.plot(val_epochs, val_top10, color=COLORS["secondary"], linewidth=2, label="Validación Top-10")
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud Top-10")
    ax.set_title("Exactitud Top-10")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "01_training_history.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:

    config = Config.from_yaml(config_path)
    device = torch.device(device_str if device_str else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # Cargar modelo
    model = create_model(config.model, config.data, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded: epoch={ckpt.get('epoch', '?')}")

    dataset = SignLanguageDataset(
        data_config=config.data,
        augmentation_config=None,
        split="test",
    )

    idx_to_label = dataset.idx_to_label
    num_classes = len(idx_to_label)
    logger.info(f"Loaded {num_classes} classes from dataset label mapping")

    loader = DataLoader(
        dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(
                batch.visual_features,
                batch.pose_features,
                batch.attention_mask,
            )
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_labels.append(batch.labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_logits = np.concatenate(all_logits)

    # Per-class metrics
    per_class = {}
    for cls in range(num_classes):
        tp = np.sum((all_preds == cls) & (all_labels == cls))
        fp = np.sum((all_preds == cls) & (all_labels != cls))
        fn = np.sum((all_preds != cls) & (all_labels == cls))
        support = int(np.sum(all_labels == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "label": idx_to_label.get(cls, f"class_{cls}"),
        }

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        conf_matrix[label, pred] += 1

    accuracy = np.mean(all_preds == all_labels)

    return {
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
        "logits": all_logits,
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "num_classes": num_classes,
        "idx_to_label": idx_to_label,
    }


def plot_confusion_matrix(
    eval_data: Dict, output_dir: Path, top_n: int = 30
) -> None:
    conf_matrix = eval_data["confusion_matrix"]
    idx_to_label = eval_data["idx_to_label"]

    # Encontrar las clases con mas errores (off-diagonal)
    errors_per_class = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    top_error_classes = np.argsort(errors_per_class)[-top_n:][::-1]

    # Sub-matrix
    sub_matrix = conf_matrix[np.ix_(top_error_classes, top_error_classes)]
    class_labels = [idx_to_label.get(c, str(c))[:12] for c in top_error_classes]

    # Normalizar por fila (recall)
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        sub_matrix.astype(float), row_sums,
        where=row_sums > 0, out=np.zeros_like(sub_matrix, dtype=float)
    )

    fig, ax = plt.subplots(figsize=(max(14, top_n * 0.5), max(12, top_n * 0.45)))

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#FFFFFF", "#DBEAFE", "#3B82F6", "#1E3A8A"]
    )
    im = ax.imshow(norm_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(class_labels, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title(f"Matriz de Confusión (Top {top_n} clases con más errores)", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Recall (normalizado por fila)")
    plt.tight_layout()

    save_path = output_dir / "02_confusion_matrix.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_worst_classes(
    eval_data: Dict, output_dir: Path, top_n: int = 50, min_support: int = 5
) -> None:
    per_class = eval_data["per_class"]

    # Filtrar clases con suficiente support
    classes_with_support = [
        (cls, m) for cls, m in per_class.items()
        if m["support"] >= min_support
    ]
    classes_with_support.sort(key=lambda x: x[1]["f1"])
    worst = classes_with_support[:top_n]

    if not worst:
        logger.warning("No classes with sufficient support for worst-class plot.")
        return

    labels = [m["label"][:20] for _, m in worst]
    f1_scores = [m["f1"] for _, m in worst]
    precisions = [m["precision"] for _, m in worst]
    recalls = [m["recall"] for _, m in worst]
    supports = [m["support"] for _, m in worst]

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.35)))

    y_pos = np.arange(len(labels))
    bar_height = 0.25

    bars_f1 = ax.barh(y_pos - bar_height, f1_scores, bar_height,
                       color=COLORS["danger"], label="F1", alpha=0.9)
    bars_p = ax.barh(y_pos, precisions, bar_height,
                      color=COLORS["primary"], label="Precisión", alpha=0.9)
    bars_r = ax.barh(y_pos + bar_height, recalls, bar_height,
                      color=COLORS["secondary"], label="Recall", alpha=0.9)

    # Anotar support
    for i, s in enumerate(supports):
        ax.text(max(f1_scores[i], precisions[i], recalls[i]) + 0.02, i,
                f"n={s}", va="center", fontsize=7, color=COLORS["neutral"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Puntuación")
    ax.set_title(f"Las {top_n} peores clases por F1 (soporte mínimo={min_support})", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    save_path = output_dir / "04_worst_classes.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_top_confusions(
    eval_data: Dict, output_dir: Path, top_n: int = 20
) -> None:
    conf_matrix = eval_data["confusion_matrix"]
    idx_to_label = eval_data["idx_to_label"]
    num_classes = eval_data["num_classes"]

    # Encontrar pares off-diagonal mas altos
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                confusions.append((i, j, conf_matrix[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = confusions[:top_n]

    if not top_confusions:
        logger.warning("No confusions found.")
        return

    pair_labels = []
    counts = []
    for true_cls, pred_cls, count in top_confusions:
        true_name = idx_to_label.get(true_cls, str(true_cls))[:15]
        pred_name = idx_to_label.get(pred_cls, str(pred_cls))[:15]
        pair_labels.append(f"{true_name} -> {pred_name}")
        counts.append(count)

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))

    colors_bar = plt.cm.Reds(np.linspace(0.3, 0.9, len(counts)))
    ax.barh(range(len(counts)), counts, color=colors_bar, edgecolor="white")

    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel("Frecuencia de Confusión")
    ax.set_title(f"Top {top_n} pares más confundidos (Real → Predicho)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Anotar counts
    for i, c in enumerate(counts):
        ax.text(c + 0.5, i, str(c), va="center", fontsize=8, color=COLORS["text"])

    plt.tight_layout()
    save_path = output_dir / "06_top_confusions.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def generate_summary_report(eval_data: Dict, output_dir: Path) -> None:
    """Genera un resumen en texto con las metricas principales."""
    per_class = eval_data["per_class"]
    accuracy = eval_data["accuracy"]

    classes_with_samples = [m for m in per_class.values() if m["support"] > 0]
    precisions = [m["precision"] for m in classes_with_samples]
    recalls = [m["recall"] for m in classes_with_samples]
    f1s = [m["f1"] for m in classes_with_samples]

    report = []
    report.append("MODEL EVALUATION SUMMARY")
    report.append(f"Overall Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    report.append(f"Number of Classes:      {eval_data['num_classes']}")
    report.append(f"Classes with Samples:   {len(classes_with_samples)}")
    report.append(f"Total Samples Evaluated:{len(eval_data['labels'])}")
    report.append("")
    report.append("Macro-Averaged Metrics:")
    report.append(f"  Precision:  {np.mean(precisions):.4f} (std={np.std(precisions):.4f})")
    report.append(f"  Recall:     {np.mean(recalls):.4f} (std={np.std(recalls):.4f})")
    report.append(f"  F1-Score:   {np.mean(f1s):.4f} (std={np.std(f1s):.4f})")
    report.append("")
    report.append("Percentiles (F1):")
    for p in [10, 25, 50, 75, 90]:
        report.append(f"  P{p}: {np.percentile(f1s, p):.4f}")
    report.append("")

    # Peores 10
    sorted_classes = sorted(classes_with_samples, key=lambda m: m["f1"])
    report.append("Worst 10 Classes by F1:")
    for m in sorted_classes[:10]:
        report.append(
            f"  {m['label']:<25s} F1={m['f1']:.4f}  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  n={m['support']}"
        )
    report.append("")

    # Mejores 10
    report.append("Best 10 Classes by F1:")
    for m in sorted_classes[-10:][::-1]:
        report.append(
            f"  {m['label']:<25s} F1={m['f1']:.4f}  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  n={m['support']}"
        )

    report_text = "\n".join(report)
    print(report_text)

    save_path = output_dir / "00_summary.txt"
    with open(save_path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved: {save_path}")


def plot_accuracy_histogram(eval_data: Dict, output_dir: Path) -> None:

    per_class = eval_data["per_class"]
    
    # Filtramos solo clases con support > 0
    accuracies = [m["recall"] * 100 for m in per_class.values() if m["support"] > 0]
    total_classes = len(accuracies)
    
    if total_classes == 0:
        return

    # Definir los rangos (bins)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]
    
    # Calcular histograma
    counts, _ = np.histogram(accuracies, bins=bins)
    
    
    print("DISTRIBUCIÓN REAL DE ACCURACY (HISTOGRAMA)")
    print("="*50)
    print(f"{'RANGO':<15} | {'CANTIDAD':<10} | {'PORCENTAJE':<10}")
    
    
    for i, count in enumerate(counts):
        pct = (count / total_classes) * 100
        print(f"{labels[i]:<15} | {count:<10d} | {pct:<6.2f}%")
        
    
    print(f"Total Clases: {total_classes}")
    

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Colores: Rojo para bajo, Amarillo medio, Verde alto
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(counts)))
    
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", alpha=0.8)
    
    # Anotar valores
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Número de Clases")
    ax.set_xlabel("Rango de Exactitud (Recall)")
    ax.set_title(f"Distribución de exactitud por clase (Total: {total_classes})", fontsize=14, fontweight='bold')
    ax.grid(True, axis="y", alpha=0.3)

    # Línea de promedio — calcular índice correcto de la barra
    avg_acc = np.mean(accuracies)
    bar_index = min(int(avg_acc // 10), len(labels) - 1)
    ax.axvline(x=bar_index, color='blue', linestyle='--', linewidth=2,
               label=f'Promedio: {avg_acc:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    save_path = output_dir / "08_accuracy_histogram.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Generate visual metrics report for sign language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="outputs/config.yaml")
    parser.add_argument("--labels", type=str, default="data/labels.json")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics_report")
    parser.add_argument("--history", type=str, default="outputs/training_history.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-confusion-classes", type=int, default=30)
    parser.add_argument("--top-worst-classes", type=int, default=30)
    parser.add_argument("--min-support", type=int, default=5)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating training history plots")
    plot_training_history(args.history, output_dir)

    logger.info("Running full evaluation on validation set")
    eval_data = evaluate_model(
        args.checkpoint, args.config, args.labels,
        args.features_dir, args.device
    )

    generate_summary_report(eval_data, output_dir)

    logger.info("confusion matrix")
    plot_confusion_matrix(eval_data, output_dir, top_n=args.top_confusion_classes)

    logger.info("Generating worst classes plot")
    plot_worst_classes(eval_data, output_dir, top_n=args.top_worst_classes, min_support=args.min_support)

    logger.info("Generating top confusions")
    plot_top_confusions(eval_data, output_dir)

    logger.info("Generating accuracy tiers analysis")
    plot_accuracy_histogram(eval_data, output_dir)

    print(f"Files generated:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()