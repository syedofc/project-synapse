# ==============================================================================
# File 07b: 07_training/07b_dashboard.py
# Purpose: A simple dashboard to display live training progress in the terminal.
# ==============================================================================
"""
A simple terminal-based dashboard for live training progress visualization.
"""
import sys

class Dashboard:
    def log_message(self, message):
        print(message)

    def log_start(self, config):
        print("="*60)
        print(f"Starting Training: {config['project_name']}")
        print(f"Device: {config['training']['device']}")
        print(f"Epochs: {config['training']['epochs']}, Batch Size: {config['training']['batch_size']}")
        print(f"Optimizer: {config['training']['optimizer']}, LR: {config['training']['learning_rate']}")
        print("="*60)

    def update(self, epoch, step, total_steps, loss, acc):
        """Updates the live progress bar for the current training epoch."""
        bar_length = 25
        progress = float(step) / total_steps
        block = int(round(bar_length * progress))
        progress_bar = f"[{'=' * block + '>' + '.' * (bar_length - block - 1)}]"
        
        # Use carriage return to overwrite the line
        text = f"\rEpoch {epoch} | {progress_bar} {step}/{total_steps} | Loss: {loss:.4f} | Acc: {acc:.2%}"
        sys.stdout.write(text)
        sys.stdout.flush()

    def log_epoch_summary(self, epoch, total_epochs, duration, train_loss, train_acc, val_loss, val_acc, val_metrics_by_task):
        """Prints a summary at the end of an epoch."""
        # Clear the progress bar line before printing the summary
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()

        print(f"Epoch {epoch}/{total_epochs} Summary | Duration: {duration:.2f}s")
        print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
        print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
        for task_name, metrics in val_metrics_by_task.items():
            metric_parts = [f"Loss: {metrics['loss']:.4f}"]
            for metric_name, metric_value in metrics.items():
                if metric_name == 'loss':
                    continue
                if isinstance(metric_value, float):
                    metric_parts.append(f"{metric_name}: {metric_value:.2%}")
                else:
                    metric_parts.append(f"{metric_name}: {metric_value}")
            print(f"    - Task '{task_name}': " + ", ".join(metric_parts))
        print("-"*60)
