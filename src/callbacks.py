from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, progress

""" Callbacks for model checkpointing and training logging """
def get_cycle_gan_callbacks(checkpoint_path, resume_training=False) -> list:

    lr_logger = LearningRateMonitor(logging_interval='epoch')
    filename = None
    if resume_training:
        filename = 'resumed_epoch={epoch}-step={step}'

    checkpoint_callback = ModelCheckpoint(
                                          dirpath=checkpoint_path,
                                          save_top_k = -1,
                                          every_n_epochs=3,
                                          filename=filename,
                                          save_last=not resume_training)
    if resume_training:
        checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last-resumed"

    callbacks = [lr_logger, checkpoint_callback]

    return callbacks