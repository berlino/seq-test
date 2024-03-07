import argparse
import random
from datetime import datetime
from typing import List, Union
import pandas as pd

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            logits = self.model(inputs)

            # collect auxiliary losses
            auxiliary_loss = []

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            self.model.apply(get_auxiliary_loss)
            auxiliary_loss = sum(auxiliary_loss)

            # need to flatten batch and sequence dimensions
            main_loss = self.loss_fn(
                rearrange(logits, "... c -> (...) c"), targets.flatten()
            )
            loss = main_loss + auxiliary_loss
            loss.backward()
            self.optimizer.step()

            # logging and printing
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log(
                {
                    "train/loss": loss,
                    "train/main_loss": main_loss,
                    "train/auxiliary_loss": auxiliary_loss,
                    "epoch": epoch_idx,
                }
            )

    def test(self, epoch_idx: int):
        self.model.eval()

        test_loss = 0
        # all_preds = []
        # all_targets = []
        results = [] 

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)

                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), targets.flatten()
                )
                test_loss += loss / len(self.test_dataloader)

                # SE: important to
                preds = torch.argmax(logits, dim=-1).cpu()
                results.extend(compute_metrics(preds, targets.cpu(), slices))
               
                iterator.update(1)

            # test_accuracy = compute_accuracy(
            #     torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
            # )
            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})
        return metrics

    def fit(self):
        self.model.to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    # TODO (SE): need to actaully verify reproducibility here
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config=config.model)
    
    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    task.fit()

    # run post-training evaluation for regbench
    # TODO: move this to a separate function
    assert len(config.data.test_configs) == 1
    if config.data.test_configs[0].name == "regbench":
        from zoology.data.regbench import regbench
        eval_input, id2token, eval_dfas = regbench(**config.data.test_configs[0].model_dump(), seed=config.data.seed, eval_flag=True)
        num_examples, seq_len = eval_input.shape
        batch_size = config.data.batch_size
        num_batches = math.ceil(num_examples // batch_size)
        pad_token = "<unk>"
        example_separator_token = "|"
        num_total_tokens, num_correct_tokens = 0, 0
        for batch_idx in range(num_batches):
            batch_input = eval_input[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(task.device)
            batch_dfa = eval_dfas[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            logits = model(batch_input)
            batch_preds = torch.argmax(logits, dim=-1)

            for inp, pred, dfa in zip(batch_input, batch_preds, batch_dfa):
                inp = [id2token[_t] for _t in inp]
                # print("input str: ", inp)
                pred = [id2token[_t] for _t in pred]
                assert len(inp) == len(pred)
                example_start_index = 0
                # input: a b c | d e f | <unk> <unk>
                for tok_idx in range(len(inp) - 1):
                    cur_token = inp[tok_idx]

                    # skip the pad token
                    if cur_token == pad_token:
                        example_start_index = tok_idx + 1
                        continue

                    # skip the last token of a sample, e.g.,c f
                    if inp[tok_idx + 1] == example_separator_token:
                        example_start_index = tok_idx + 2
                        continue

                    # cur token could be b c or |. In the case of |, model predits the first character of a new sample
                    dfa_str = inp[example_start_index : tok_idx + 1] + [pred[tok_idx]]
                    correctness = dfa(dfa_str)

                    num_total_tokens += 1
                    num_correct_tokens += correctness
        print(f"DFA Accuracy: {num_correct_tokens / num_total_tokens}")

    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
