from pathlib import Path

def get_config():
    """Returns default configuration parameters for training and saving the transformer model.

    Includes hyperparameters, language settings, file paths, and experiment tracking options.

    Returns:
        dict: Dictionary containing configuration values.
    """
    return {
        'batch_size': 8,
        'num_epochs': 50,
        'lr': 1e-4,
        'seq_len': 800,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'es',
        'model_folder': 'weights',
        'model_filename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch):
    """Generates the file path for saving or loading model weights for a given epoch.

    Args:
        config (dict): Configuration dictionary containing model folder and filename base.
        epoch (int): Epoch number to include in the filename.

    Returns:
        str: Full path to the model weights file.
    """

    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}{epoch}.pt"
    return str(Path(model_folder) / model_filename)
