import os
from .constant import SUPPORTED_MODELS, SPLITS, BASE_DIR

def check_supported_models(param_name, value):
    if value not in SUPPORTED_MODELS:
        raise ValueError(f"Invalid argument {param_name}='{value}'. Valid options: {SUPPORTED_MODELS}")

def check_splits(param_name, value):
    if value not in SPLITS:
        raise ValueError(f"Invalid argument {param_name}='{value}'. Valid options: {SPLITS}")

def get_model_weights_path(model_name):
    check_supported_models('model_name', model_name)
    save_dir = os.path.join(BASE_DIR, 'model_weights')
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f'{model_name}.bin')

def get_adv_images_path(model_name, psnr, split, batch_idx):
    check_supported_models('model_name', model_name)
    check_splits('split', split)
    save_dir = os.path.join(BASE_DIR, 'adv_images', model_name, f'psnr_{psnr}')
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f'{split}_batch_{batch_idx:04d}.pt')

def get_clean_features_path(model_name, split, sigma=None):
    check_supported_models('model_name', model_name)
    check_splits('split', split)
    save_dir = os.path.join(BASE_DIR, 'features', 'clean', model_name)
    os.makedirs(save_dir, exist_ok=True)
    if sigma is None:
        filename = f"{split}.h5"
    else:
        filename = f"{split}_sigma{sigma}.h5"
    return os.path.join(save_dir, filename)

def get_adv_features_path(target_model_name, source_model_name, split, psnr, sigma=None):
    check_supported_models('target_model_name', target_model_name)
    check_supported_models('source_model_name', source_model_name)
    check_splits('split', split)
    save_dir = os.path.join(BASE_DIR, 'features', 'adversarial', target_model_name, f'psnr_{psnr}')
    os.makedirs(save_dir, exist_ok=True)

    if target_model_name == source_model_name:
        if sigma is None:
            filename = f"{split}_wb.h5"
        else:
            filename = f"{split}_wb_sigma{sigma}.h5"
    else:
        if sigma is None:
            filename = f"{split}_transf_{source_model_name}.h5"
        else:
            filename = f"{split}_transf_{source_model_name}_sigma{sigma}.h5"
    return os.path.join(save_dir, filename)