import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import json
import torch

from src.model_cycle_gan_cmip import CycleGAN, DataModule
from src.utils import get_version, set_environment, get_uuid_from_path, get_checkpoint_path, get_config, save_config
from src.callbacks import get_cycle_gan_callbacks


""" Initializes and runs the GAN training.

    The three main training options via commandline are:

        1. (default) Initialize and train new model from a given configuration
        2. (--checkpoint_path) continue training with the stored model configuration
        3. (--checkpoint_path and --transfer_learning) continue training with the stored model
           using a new configuration
"""
class Training():


    def __init__(self, config):

        self.config = config
        self.version = get_version()
        set_environment()
        save_config(config, self.version)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_training = False
        self.transfer_learning = False


    """ Initialize the model """
    def initialize(self, checkpoint_path=None, transfer_learning=False):

        self.tb_logger = TensorBoardLogger(self.config.tensorboard_path,
                                           name=self.config.model_name,
                                           default_hp_metric=False,
                                           version = self.version)

        if checkpoint_path is None:
            print('initializing new model')

            self.model = CycleGAN(epoch_decay =self.config.epochs // 2,
                                  num_resnet_layer=self.config.num_resnet_layer,
                                  discriminator_layer=self.config.discriminator_layer
                                  )
            self.checkpoint_path = get_checkpoint_path(self.config, self.version)

            print(f'Checkpoint path: {self.checkpoint_path}')
            print(json.dumps(self.config.__dict__))

        else:
            print('loading model from checkpoint')
            self.checkpoint_path = checkpoint_path
            self.model = self.load_model(self.checkpoint_path)

            if transfer_learning is False:
                print('Continuing training from old checkpoint and configuration')
                self.resume_training = True

                config_path = self.config.config_path 
                uuid = get_uuid_from_path(self.checkpoint_path)
                config_loaded = get_config(uuid, path=config_path)
                self.config.epochs = config_loaded['epochs']
                print(f'Checkpoint path: {config_loaded["checkpoint_path"]}')
                print(json.dumps(config_loaded))

            if transfer_learning is True:
                print('Continuing training from old checkpoint and NEW configuration')
                self.resume_training = False
                self.transfer_learning = True
                self.checkpoint_path = get_checkpoint_path(self.config, self.version)
                print(f'Checkpoint path: {self.checkpoint_path}')

                config_path = self.config.config_path 
                uuid = get_uuid_from_path(self.checkpoint_path)
                config_loaded = get_config(uuid, path=config_path)
                self.config.num_resnet_layer = config_loaded['num_resnet_layer']
                print(json.dumps(self.config.__dict__))
            
        self.datamodule = DataModule(self.config,
                                     train_batch_size = self.config.train_batch_size,
                                     test_batch_size = self.config.test_batch_size)

        self.trainer = pl.Trainer(gpus = 1,
                                  max_epochs = self.config.epochs,
                                  precision = 16, 
                                  callbacks = get_cycle_gan_callbacks(self.checkpoint_path,
                                                                resume_training=self.resume_training),
                                  num_sanity_val_steps = 1,
                                  logger = self.tb_logger,
                                  log_every_n_steps = self.config.log_every_n_steps,
                                  deterministic = False)


    """ Train the model """
    def fit(self):

        if self.resume_training:
            self.trainer.fit(self.model, self.datamodule,
                            ckpt_path=self.checkpoint_path)
            print('Continued training finished.')
        else:
            self.trainer.fit(self.model, self.datamodule)

            print('Training finished.')

     
    """ Load the model from a given checkpoint path """
    def load_model(self, checkpoint_path):
        model = CycleGAN().load_from_checkpoint(checkpoint_path=checkpoint_path)
        model = model.to(self.device)
        return model