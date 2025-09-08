import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint callback
from src.dataloader.datamodule import PINNDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from src.model.model import PINN  # Import the PINN class from your model
from src.model.base_models import MLP
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Solver:
    def __init__(self, args):
        self.args = args
        self.name = args.wandb_name
        self.project = args.wandb_project
        print(f"Running experiment: {self.name} in project: {self.project}")
        self.validation_freq = args.validation_freq
        self.args.device = device

        if self.validation_freq > args.epochs:
            print(f"Validation frequency set to {self.validation_freq} epochs.")
            self.validation_freq = args.epochs

        self.dirpath = 'outputs/runs/' + str(self.name)
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self._params(args)
        self._load_data(args)

        # self._load_data(args)
        self._load_trainer()
    
    def _params(self, args):
        self.bs = args.batch_size
        self.epochs = args.epochs
        self.data_dir = 'data/simulation_1.pt'

    def _load_data(self, args):
        # Create the training data loader
        # Data preparation
        self.data_module = PINNDataModule(dataset_dir=self.data_dir, batch_size=args.batch_size, ratio=args.ratio)
        # Setup the data module in order to compute normalization
        args.stats = self.data_module.setup() 
        # Set seed
        pl.seed_everything(args.seed, workers=True)

        # Model instantiation
        self.base_model = MLP(input_size=args.input_dim,
                              n_hidden=args.n_hidden,
                              dim_hidden=args.dim_hidden,
                              output_size=args.output_dim)
        self.model = PINN(model=self.base_model, args=args)

        # Callbacks
        self.wandb_logger = WandbLogger(name=self.name, project=self.project)
        
        
    def _load_trainer(self):
        # Define the checkpoint callback to save the model weights
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.dirpath,  # Directory to save the model weights
            filename='best_model',  # Filename format
            save_top_k=1,  # Save only the best model
            monitor='val_loss',  # Metric to monitor for saving
            mode='min',  # Mode to select the best model (minimizing validation loss)
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')  # Monitor learning rate  

        # Set up the PyTorch Lightning Trainer 
        self.trainer = pl.Trainer(
            num_sanity_val_steps=0,
            max_epochs=self.epochs,
            logger=self.wandb_logger,
            check_val_every_n_epoch = self.validation_freq,  # Validate every n epochs
            accelerator='auto', 
            devices='auto',
            callbacks=[checkpoint_callback ,lr_monitor],  
        )

    def train(self):
        # Train the model using the Trainer
        self.trainer.fit(self.model, datamodule=self.data_module)

    def test(self):
        print("Testing the model...")
        # Test the model using the Trainer
        checkpoint_path = self.dirpath + '/best_model.ckpt'
        model = PINN.load_from_checkpoint(checkpoint_path=checkpoint_path, model=self.base_model, args=self.args)
        self.trainer.test(model, datamodule=self.data_module)
        
