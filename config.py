import os

"""

    Yapılan deneyler
        main 
            /train_104
        learning curve
            /train_104_500
            /train_104_1000
            /train_104_1500
        scibert tr
            /train_105
        scibert eng
            /train_106
        
        
        
    comet 
        ulakbim-cic-colab-v101-learning-curve       -> learning curve 
        ulakbim-cic-colab-v101-scibert-base-eng     -> scibert eng traning (freeze backbone)
        ulakbim-cic-colab-v101-scibert-base         -> scibert tr traning (freeze backbone)
"""

class Config:
    # ==========================================================================
    # 1. TEMEL AYARLAR (SABİT)
    # ==========================================================================
    MODELS = [
        "dbmdz/bert-base-turkish-cased",
        "dbmdz/electra-base-turkish-cased-discriminator",
        "xlm-roberta-base",
        "microsoft/deberta-v3-base",
        "answerdotai/ModernBERT-base",
        "allenai/scibert_scivocab_uncased"      # SciBERT eklendi
    ]

    COMET_API_KEY = "LrkBSXNSdBGwikgVrzE2m73iw"
    COMET_WORKSPACE = "kemalsami"
    COMET_ONLINE_MODE = True

    # Eğitim Parametreleri
    NUMBER_TRIALS = 20
    NUMBER_EPOCHS = 50
    PATIENCE = 10
    NUMBER_CPU = 0
    SEED = 42
    BATCH_SIZE = 16
    MAX_LEN = 128
    LR_WARMUP_RATIO = 0.1
    # Seçenekler: "FocalLoss", "CrossEntropyLoss" , "CrossEntropyLoss_Weighted"
    LOSS_FUNCTION = "FocalLoss"
    # Seçenekler: "macro_f1", "accuracy"
    EVALUATION_METRIC = "accuracy"

    DATA_DIR = "data"

    # ==========================================================================
    # 2. DENEY YAPILANDIRMASI (EXPERIMENT ID)
    # ==========================================================================
    EXPERIMENT_ID = 1
    MODEL_INDEX = 0
    BACKGROUND_THRESHOLD = 0.5
    TRAIN_SIZE = None  # None, 500, 1000, 1500

    ACTIVE_MODEL_NAME = None
    PREFIX_DIR = "_train_104/"
    CONTEXT_RICH = False

    # Dinamik Yollar (set_experiment ve set_model ile dolar)
    CHECKPOINT_DIR = ""
    RESULTS_DIR = ""
    DATA_PATH_TRAIN = ""
    DATA_PATH_VAL = ""
    DATA_PATH_TEST = ""

    # Comet İsimlendirme
    COMET_PROJECT_PREFIX = ""   # örn: experiment-1
    COMET_PROJECT_NAME = ""     # örn: experiment-1-bert-base

    @classmethod
    def should_freeze_backbone(cls):
        """
        Aktif model SciBERT ise True, değilse False döndürür.
        İleride başka modelleri dondurmak istersen buraya ekleyebilirsin.
        """
        if cls.ACTIVE_MODEL_NAME and "scibert" in cls.ACTIVE_MODEL_NAME.lower():
            return True
        return False


    @classmethod
    def _setup_paths(cls):
        """EXPERIMENT_ID değerine göre klasör ve veri yollarını ayarlar."""
        exp_id = int(cls.EXPERIMENT_ID)

        # 1) Klasör İsimlendirmeleri (checkpoints_v1, checkpoints_v2, ...)
        # İsteğinize uygun olarak 'v' yanına direkt ID geliyor.
        cls.CHECKPOINT_DIR = f"{cls.PREFIX_DIR}checkpoints_v{exp_id}"

        # 2) Sonuç Klasörü ve Comet Proje İsimleri
        # Her deneyin çıktısı farklı klasöre gider
        result_folders = {
            1: "experiment_1_flat",
            2: "experiment_2_hierarchical",
            3: "experiment_3_hierarchical_rich",
            4: "experiment_4_flat_rich",
            5: "experiment_5_flat_ext_no_section"
        }
        folder_name = result_folders.get(exp_id, f"experiment_{exp_id}_custom")
        cls.RESULTS_DIR = os.path.join(f"{cls.PREFIX_DIR}outputs", folder_name)

        # 3) Veri Yolları
        if exp_id in [3, 4, 5]:
            suffix = "_ext"
            cls.MAX_LEN = 256  # Context olduğu için daha uzun
        else:
            suffix = ""  # Exp 1 ve 2 normal veri
            cls.MAX_LEN = 128

        # Sadece experiment 3 ve 4 için section bilgisi alınabilir
        if exp_id in [3, 4]:
            cls.CONTEXT_RICH = True
        else:
            cls.CONTEXT_RICH = False

        if cls.TRAIN_SIZE:
            cls.DATA_PATH_TRAIN = os.path.join(cls.DATA_DIR, f"data_v2_train_{cls.TRAIN_SIZE}.csv")
        else:
            cls.DATA_PATH_TRAIN = os.path.join(cls.DATA_DIR, f"data_v2_train{suffix}.csv")

        cls.DATA_PATH_VAL = os.path.join(cls.DATA_DIR, f"data_v2_val{suffix}.csv")
        cls.DATA_PATH_TEST = os.path.join(cls.DATA_DIR, f"data_v2_test{suffix}.csv")

        # 4) Comet Prefix (Model isminden bağımsız kısım)
        cls.COMET_PROJECT_PREFIX = f"experiment-{exp_id}"
        if cls.TRAIN_SIZE:
            cls.COMET_PROJECT_PREFIX += f"-size{cls.TRAIN_SIZE}"

        cls.ensure_directories()

    @classmethod
    def set_prefix(cls, prefix_dir):
        """Prefix değerini ayarlar. None gelirse değiştirmez."""
        # args parametresi olarak verilmemişse default değer alınır
        if prefix_dir is None:
            prefix_dir = cls.PREFIX_DIR

        cls.PREFIX_DIR = str(prefix_dir)
        cls._setup_paths()  # Prefix değişince yolları güncelle

    @classmethod
    def set_experiment(cls, experiment_id):
        # args parametresi olarak verilmemişse default değer alınır
        if experiment_id is None:
            experiment_id = cls.EXPERIMENT_ID

        cls.EXPERIMENT_ID = int(experiment_id)
        cls._setup_paths()
        if cls.ACTIVE_MODEL_NAME:
            cls._update_comet_name()

    @classmethod
    def set_train_size(cls, size):
        """Eğitim veri seti boyutunu ayarlar (500, 1000, 1500 vb). None ise tam set."""
        if size is not None:
            cls.TRAIN_SIZE = int(size)
        else:
            cls.TRAIN_SIZE = None
        cls._setup_paths()
        if cls.ACTIVE_MODEL_NAME:
            cls._update_comet_name()

    @classmethod
    def set_model(cls, model_index):
        # args parametresi olarak verilmemişse default değer alınır
        if model_index is None:
            model_index = cls.MODEL_INDEX

        idx = int(model_index)
        if not (0 <= idx < len(cls.MODELS)):
            raise IndexError(f"Geçersiz model indeksi: {idx}. (0-{len(cls.MODELS) - 1} arası olmalı)")

        cls.MODEL_INDEX = idx
        cls.ACTIVE_MODEL_NAME = cls.MODELS[idx]
        cls._update_comet_name()

    @classmethod
    def _update_comet_name(cls):
        """Comet proje ismini (Prefix + Model Short Name) birleştirir."""
        if cls.ACTIVE_MODEL_NAME and cls.COMET_PROJECT_PREFIX:
            short_name = cls.get_model_short_name(cls.ACTIVE_MODEL_NAME)
            cls.COMET_PROJECT_NAME = f"{cls.COMET_PROJECT_PREFIX}-{short_name}"

    @classmethod
    def get_model_short_name(cls, model_name):
        return model_name.split('/')[-1]

    @classmethod
    def get_checkpoint_path(cls, model_name=None):
        name = model_name if model_name else cls.ACTIVE_MODEL_NAME
        if not name:
            raise ValueError("Model seçili değil!")
        short_name = cls.get_model_short_name(name)
        return os.path.join(cls.CHECKPOINT_DIR, short_name)

    @classmethod
    def get_optuna_db_path(cls, model_name=None):
        name = model_name if model_name else cls.ACTIVE_MODEL_NAME
        if not name:
            raise ValueError("Model seçili değil!")
        short_name = cls.get_model_short_name(name)
        return os.path.join(cls.CHECKPOINT_DIR, f"{short_name}.db")

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 60)
        print(f"{'AKTİF KONFIGÜRASYON':^60}")
        print("=" * 60)
        print(f" Experiment ID     : {cls.EXPERIMENT_ID}")
        print(f" Model Index       : {cls.MODEL_INDEX}")
        print(f" Train Size        : {cls.TRAIN_SIZE if cls.TRAIN_SIZE else 'FULL'}")
        print(f" Aktif Model       : {cls.ACTIVE_MODEL_NAME if cls.ACTIVE_MODEL_NAME else 'SEÇİLMEDİ'}")
        print(f" Comet Project     : {cls.COMET_PROJECT_NAME if cls.COMET_PROJECT_NAME else '---'}")
        print("-" * 60)
        print(f" Checkpoint Dir    : {cls.CHECKPOINT_DIR}")
        print(f" Results Dir       : {cls.RESULTS_DIR}")
        print("-" * 60)
        print(f" Train Data        : {cls.DATA_PATH_TRAIN}")
        print(f" Val Data          : {cls.DATA_PATH_VAL}")
        print(f" Test Data         : {cls.DATA_PATH_TEST}")
        print("-" * 60)
        print(f" Context Rich      : {cls.CONTEXT_RICH}")
        print(f" Max Token Length  : {cls.MAX_LEN}")
        print("=" * 60 + "\n")


# Varsayılan kurulum
Config._setup_paths()