import os

BASE_DIR = '/content/drive/MyDrive/fsaa_pfms'

SEED = 42
SPLITS = ['train', 'valid', 'test']
SELECTED_CLASSES = [
    'Bladder_Urothelial_Carcinoma',
    'Colon_Rectum_adenocarcinoma',
    'Prostate_adenocarcinoma',
    'Skin_Cutaneous_Melanoma',
    'Thyroid_carcinoma',
]
NUM_CLASSES = len(SELECTED_CLASSES)
LABELS_TO_IDS = {label: i for i, label in enumerate(sorted(SELECTED_CLASSES))}
IDS_TO_LABELS = {i: label for label, i in LABELS_TO_IDS.items()}
PATTERNS = {
    "train": [os.path.join(BASE_DIR, f"data/dataset_internal_train_part{str(i).zfill(3)}.tar") for i in range(39)],
    "valid": [os.path.join(BASE_DIR, f"data/dataset_internal_valid_part{str(i).zfill(3)}.tar") for i in range(9)],
    "test": [os.path.join(BASE_DIR,f"data/dataset_internal_test_part{str(i).zfill(3)}.tar") for i in range(9)],
}

MODEL_TO_REPO = {
    'uni_v1': 'MahmoodLab/UNI',
    'conch_v1': 'MahmoodLab/CONCH',
    'virchow_v1': 'paige-ai/Virchow',
}
SUPPORTED_MODELS = list(MODEL_TO_REPO.keys())
MODEL2EMB_DIM = {
    'uni_v1': 1024,
    'conch_v1': 512,
    'virchow_v1': 2560
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MSE_TARGETS = {
    "45dB": 10**(-45 / 10),
    "40dB": 10**(-40 / 10),
    "35dB": 10**(-35 / 10)
}

SIGMA = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]