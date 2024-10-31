import torch as t
import wandb
from utils import TransformerTrainingArgs, get_log_probs, get_adaptive_optimizer
from transformer_modules import DemoTransformer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, dataset_dict):
        super().__init__()
        self.model = model
        self.args = args
        self.dataset_dict = dataset_dict
        if model.cfg.apply_UmuP:
            self.optimizer = get_adaptive_optimizer(model, args.lr)
        else:
            self.optimizer = t.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0


    def training_step(self, batch):
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss


    def validation_step(self, batch):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for
        the whole validation set).
        '''
        tokens = batch["tokens"].to(device)
        logits: Tensor = self.model(tokens)[:, :-1]
        predicted_tokens = logits.argmax(dim=-1)
        correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()
        return correct_predictions


    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total = self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i >= self.args.max_steps_per_epoch:
                    break

            correct_predictions = t.concat([self.validation_step(batch) for batch in self.test_loader()])
            accuracy = correct_predictions.float().mean().item()
            wandb.log({"accuracy": accuracy}, step=self.step)

        wandb.finish()


    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(self.dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(self.dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)