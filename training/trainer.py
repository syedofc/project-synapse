# File: training/trainer.py (Final Correction for Caching Call)
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from itertools import cycle

from . import dashboard 
from utils import metrics as metrics_utils 

class Trainer:
    def __init__(self, model, config, train_loaders, test_loaders):
        self.model = model
        self.config = config
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.tasks_config = config['tasks']
        
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.get_trainable_parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.loss_fns = {}
        for task_name, task_def in self.tasks_config.items():
            if task_name not in self.train_loaders: 
                if task_name in self.test_loaders: 
                    del self.test_loaders[task_name]
                continue
            loss_type = task_def.get('loss_function', 'CrossEntropyLoss')
            if loss_type == 'CrossEntropyLoss':
                self.loss_fns[task_name] = nn.CrossEntropyLoss()
            elif loss_type == 'DiceLoss':
                print(f"Warning: DiceLoss for task {task_name} placeholder in Trainer. Using CrossEntropyLoss.")
                self.loss_fns[task_name] = nn.CrossEntropyLoss() 
            else:
                raise ValueError(f"Unsupported loss function type: {loss_type} for task {task_name}")
        
        self.dashboard = dashboard.Dashboard()
        self.best_overall_val_metric = 0.0 
        self.primary_metric_is_accuracy = True 

        self.scheduler = None
        if 'learning_rate_scheduler' in config['training'] and config['training']['learning_rate_scheduler']:
            if config['training']['learning_rate_scheduler'] == 'StepLR':
                step_size = config['training'].get('scheduler_step_size', 30)
                gamma = config['training'].get('scheduler_gamma', 0.1)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            else:
                print(f"Warning: Unknown learning rate scheduler: {config['training']['learning_rate_scheduler']}")

    def train(self):
        epochs = self.config['training']['epochs']
        self.dashboard.log_start(self.config)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            train_loss, train_metrics_agg = self._train_one_epoch(epoch)
            val_loss, val_metrics_agg, val_metrics_by_task = self._evaluate(epoch)
            
            if self.scheduler:
                self.scheduler.step()
            
            duration = time.time() - epoch_start_time
            current_val_metric_for_comparison = val_metrics_agg.get('accuracy', 0.0) 

            self.dashboard.log_epoch_summary(
                epoch, epochs, duration, 
                train_loss, train_metrics_agg.get('accuracy', 0.0), 
                val_loss, current_val_metric_for_comparison, 
                val_metrics_by_task
            )
            
            if current_val_metric_for_comparison > self.best_overall_val_metric:
                self.best_overall_val_metric = current_val_metric_for_comparison
                checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, f"{self.config['project_name']}_best_model_epoch_{epoch}_val_metric_{self.best_overall_val_metric:.4f}.pth")
                try:
                    torch.save(self.model.state_dict(), save_path)
                    self.dashboard.log_message(f"Saved new best model to {save_path}")
                except Exception as e:
                    self.dashboard.log_message(f"Error saving model: {e}")

    def evaluate(self, epoch=0):
        self.dashboard.log_message("Starting evaluation...")
        val_loss, val_metrics_agg, val_metrics_by_task = self._evaluate(epoch)
        self.dashboard.log_message(
            f"Evaluation complete | Loss: {val_loss:.4f} | Accuracy: {val_metrics_agg.get('accuracy', 0.0):.2%}"
        )
        for task_name, metrics in val_metrics_by_task.items():
            self.dashboard.log_message(f"  - {task_name}: {metrics}")
        return val_loss, val_metrics_agg, val_metrics_by_task

    def _train_one_epoch(self, epoch):
        self.model.train() 
        total_loss_agg = 0.0
        total_correct_agg = 0
        total_samples_agg = 0
        processed_steps = 0
        
        active_task_names = [name for name in self.tasks_config.keys() if name in self.train_loaders]
        if not active_task_names:
            self.dashboard.log_message("Warning: No active tasks with data loaders for training in _train_one_epoch.")
            return 0.0, {'accuracy': 0.0}

        iters = {name: cycle(self.train_loaders[name]) for name in active_task_names}
        num_batches_per_main_task_cycle = max(len(self.train_loaders[name]) for name in active_task_names) if active_task_names else 0
        total_steps_for_epoch = num_batches_per_main_task_cycle * len(active_task_names)
        
        if total_steps_for_epoch == 0:
             self.dashboard.log_message("Warning: total_steps_for_epoch is 0. No training batches to process.")
             return 0.0, {'accuracy': 0.0}

        task_names_iterator = cycle(active_task_names)

        for step_idx in range(total_steps_for_epoch):
            task_name = next(task_names_iterator)
            task_id_scalar = self.tasks_config[task_name]['id']
            
            try:
                batch_data = next(iters[task_name])
            except StopIteration:
                self.dashboard.log_message(f"Warning: Dataloader for {task_name} exhausted unexpectedly.")
                continue

            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                inputs_raw, current_labels_raw = batch_data
                if isinstance(inputs_raw, dict):
                    current_inputs_dict_raw = inputs_raw
                elif torch.is_tensor(inputs_raw):
                    current_inputs_dict_raw = {'camera_image': inputs_raw} 
                else:
                    self.dashboard.log_message(f"ERROR: Task '{task_name}' data loader yielded unexpected input format: {type(inputs_raw)}. Skipping batch.")
                    continue
            else:
                self.dashboard.log_message(f"ERROR: Task '{task_name}' data loader yielded unexpected batch_data format: {type(batch_data)}. Expected (inputs, labels). Skipping batch.")
                continue
            
            inputs_dict_on_device = {}
            try:
                for key, val_tensor in current_inputs_dict_raw.items():
                    if torch.is_tensor(val_tensor):
                        inputs_dict_on_device[key] = val_tensor.to(self.device)
                    else: 
                        inputs_dict_on_device[key] = val_tensor 
                labels_on_device = current_labels_raw.to(self.device) if torch.is_tensor(current_labels_raw) else current_labels_raw
            except Exception as e:
                self.dashboard.log_message(f"Error moving data to device for task {task_name}: {e}. Skipping batch.")
                continue

            self.optimizer.zero_grad()
            outputs = self.model(inputs_dict_on_device, task_id_scalar)
            
            current_loss_fn = self.loss_fns[task_name]
            loss = current_loss_fn(outputs, labels_on_device) 

            loss.backward()
            self.optimizer.step()
            
            total_loss_agg += loss.item()
            processed_steps += 1
            
            if self.tasks_config[task_name].get('metrics', ['accuracy'])[0] == 'accuracy':
                 acc_batch = metrics_utils.calculate_accuracy(outputs, labels_on_device)
                 total_correct_agg += acc_batch * labels_on_device.size(0)
                 total_samples_agg += labels_on_device.size(0)

            if not self.model.last_used_from_cache: 
                self.model.cache_current_state() # <<< CORRECTED CALL: No arguments needed

            current_primary_metric = (total_correct_agg / total_samples_agg) if total_samples_agg > 0 else 0.0
            self.dashboard.update(epoch, step_idx + 1, total_steps_for_epoch, loss.item(), current_primary_metric)
            
        avg_loss = total_loss_agg / processed_steps if processed_steps > 0 else 0.0
        avg_primary_metric = (total_correct_agg / total_samples_agg) if total_samples_agg > 0 else 0.0
        
        return avg_loss, {'accuracy': avg_primary_metric}

    def _evaluate(self, epoch): # Logic for evaluation remains same as previous correct version
        self.model.eval() 
        total_loss_agg = 0.0
        total_correct_agg = 0
        total_samples_agg = 0
        metrics_by_task = {}
        total_batches_evaluated = 0

        with torch.no_grad():
            for task_name, loader in self.test_loaders.items():
                if task_name not in self.tasks_config or task_name not in self.loss_fns: 
                    self.dashboard.log_message(f"Warning: Task '{task_name}' missing config or loss_fn during eval. Skipping.")
                    continue

                task_id_scalar = self.tasks_config[task_name]['id']
                current_loss_fn = self.loss_fns[task_name]
                
                task_total_loss_for_log = 0.0
                task_total_correct_for_log = 0
                task_total_samples_for_log = 0
                task_batches_evaluated = 0
                
                for batch_data in loader: 
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        inputs_raw, current_labels_raw = batch_data
                        if isinstance(inputs_raw, dict):
                            current_inputs_dict_raw = inputs_raw
                        elif torch.is_tensor(inputs_raw):
                            current_inputs_dict_raw = {'camera_image': inputs_raw}
                        else:
                            self.dashboard.log_message(f"ERROR: Task '{task_name}' (eval) data loader yielded unexpected input format: {type(inputs_raw)}. Skipping batch.")
                            continue
                    else:
                        self.dashboard.log_message(f"ERROR: Task '{task_name}' (eval) data loader yielded unexpected batch_data format: {type(batch_data)}. Skipping batch.")
                        continue

                    inputs_dict_on_device = {}
                    try:
                        for key, val_tensor in current_inputs_dict_raw.items():
                            if torch.is_tensor(val_tensor):
                                inputs_dict_on_device[key] = val_tensor.to(self.device)
                            else:
                                inputs_dict_on_device[key] = val_tensor
                        labels_on_device = current_labels_raw.to(self.device) if torch.is_tensor(current_labels_raw) else current_labels_raw
                    except Exception as e:
                        self.dashboard.log_message(f"Error moving eval data to device for task {task_name}: {e}. Skipping batch.")
                        continue
                    
                    outputs = self.model(inputs_dict_on_device, task_id_scalar)
                    loss = current_loss_fn(outputs, labels_on_device)
                    
                    task_total_loss_for_log += loss.item()
                    total_loss_agg += loss.item() 
                    total_batches_evaluated +=1
                    task_batches_evaluated += 1
                    
                    if self.tasks_config[task_name].get('metrics', ['accuracy'])[0] == 'accuracy':
                        acc_batch = metrics_utils.calculate_accuracy(outputs, labels_on_device)
                        task_total_correct_for_log += acc_batch * labels_on_device.size(0)
                        total_correct_agg += acc_batch * labels_on_device.size(0) 
                    
                    task_total_samples_for_log += labels_on_device.size(0)
                    total_samples_agg += labels_on_device.size(0)
                
                avg_task_loss = task_total_loss_for_log / task_batches_evaluated if task_batches_evaluated > 0 else 0.0
                avg_task_primary_metric = (task_total_correct_for_log / task_total_samples_for_log) if task_total_samples_for_log > 0 else 0.0
                
                metrics_by_task[task_name] = {'loss': avg_task_loss}
                if self.tasks_config[task_name].get('metrics', ['accuracy'])[0] == 'accuracy':
                     metrics_by_task[task_name]['acc'] = avg_task_primary_metric
                elif self.tasks_config[task_name].get('metrics', ['accuracy'])[0] == 'mIoU': 
                     metrics_by_task[task_name]['mIoU'] = avg_task_primary_metric

        avg_overall_loss = total_loss_agg / total_batches_evaluated if total_batches_evaluated > 0 else 0.0
        avg_overall_primary_metric = (total_correct_agg / total_samples_agg) if total_samples_agg > 0 else 0.0
        
        return avg_overall_loss, {'accuracy': avg_overall_primary_metric}, metrics_by_task
