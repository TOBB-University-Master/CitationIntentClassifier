import os


class Config:
    # ==========================================================================
    # 1. TEMEL AYARLAR (SABİT)
    # ==========================================================================
    MODELS = [
        "dbmdz/bert-base-turkish-cased",
        "dbmdz/electra-base-turkish-cased-discriminator",
        "xlm-roberta-base",
        "microsoft/deberta-v3-base",
        "answerdotai/ModernBERT-base"
    ]

    COMET_API_KEY = "LrkBSXNSdBGwikgVrzE2m73iw"
    COMET_WORKSPACE = "ulakbim-cic-colab-v101"
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
    LOSS_FUNCTION = "CrossEntropyLoss"
    # Seçenekler: "macro_f1", "accuracy"
    EVALUATION_METRIC = "accuracy"

    DATA_DIR = "data"

    # ==========================================================================
    # 2. DENEY YAPILANDIRMASI (EXPERIMENT ID)
    # ==========================================================================
    # Buradaki değeri değiştirerek (1, 2 veya 3) tüm ayarları güncelleyebilirsiniz.
    EXPERIMENT_ID = 1

    # Başlangıçta boş tanımlıyoruz, _setup() metodu bunları dolduracak
    PREFIX_DIR = ""
    CHECKPOINT_DIR = ""
    RESULTS_DIR = ""
    DATA_PATH_TRAIN = ""
    DATA_PATH_VAL = ""
    DATA_PATH_TEST = ""
    COMET_PROJECT_NAME_PREFIX = ""

    @classmethod
    def _setup(cls):
        """EXPERIMENT_ID değerine göre tüm yolları dinamik olarak ayarlar."""
        exp_id = int(cls.EXPERIMENT_ID)

        # 1) Klasör İsimlendirmeleri (checkpoints_v1, checkpoints_v2, ...)
        # İsteğinize uygun olarak 'v' yanına direkt ID geliyor.
        cls.CHECKPOINT_DIR = f"{cls.PREFIX_DIR}checkpoints_v{exp_id}"

        # 2) Sonuç Klasörü ve Comet Proje İsimleri
        # Her deneyin çıktısı farklı klasöre gider
        result_folders = {
            1: "experiment_1_flat",
            2: "experiment_2_hierarchical",
            3: "experiment_3_context_aware"
        }
        folder_name = result_folders.get(exp_id, f"experiment_{exp_id}_custom")
        cls.RESULTS_DIR = os.path.join(f"{cls.PREFIX_DIR}outputs", folder_name)

        # Comet Proje İsmi: experiment-1-flat, experiment-2-hierarchical vb.
        comet_suffixes = {
            1: "flat",
            2: "hierarchical",
            3: "rich"
        }
        suffix = comet_suffixes.get(exp_id, "custom")
        cls.COMET_PROJECT_NAME_PREFIX = f"experiment-{exp_id}-{suffix}"

        # 3) Veri Dosyası Suffix (Uzantı) Mantığı
        # Exp 1 -> Normal (data_v2_train.csv)
        # Exp 2 -> Augmented (data_v2_train_aug.csv), ama Test normal (data_v2_test.csv)
        # Exp 3 -> Extended (data_v2_train_ext.csv), Test de Extended (data_v2_test_ext.csv)

        if exp_id == 3:
            # Context-Aware: Tüm dosyalar _ext uzantılı
            suffix_train = "_ext"
            suffix_test = "_ext"
        elif exp_id == 2:
            # Hierarchical: Eğitim aug, Test normal (Yapısı gereği)
            suffix_train = ""
            suffix_test = ""
        else:
            # Flat / Default: Hepsi normal
            suffix_train = ""
            suffix_test = ""

        cls.DATA_PATH_TRAIN = os.path.join(cls.DATA_DIR, f"data_v2_train{suffix_train}.csv")
        cls.DATA_PATH_VAL = os.path.join(cls.DATA_DIR, f"data_v2_val{suffix_train}.csv")
        cls.DATA_PATH_TEST = os.path.join(cls.DATA_DIR, f"data_v2_test{suffix_test}.csv")

        # Klasörlerin varlığını garantiye al
        cls.ensure_directories()

    @classmethod
    def set_experiment(cls, experiment_id):
        """Dışarıdan deney ID'sini değiştirmek için kullanılır."""
        cls.EXPERIMENT_ID = int(experiment_id)
        cls._setup()
        print(f"✅ Deney Ayarlandı: {cls.EXPERIMENT_ID} -> {cls.CHECKPOINT_DIR}")

    @classmethod
    def get_model_short_name(cls, model_name):
        return model_name.split('/')[-1]

    @classmethod
    def get_checkpoint_path(cls, model_name):
        short_name = cls.get_model_short_name(model_name)
        return os.path.join(cls.CHECKPOINT_DIR, short_name)

    @classmethod
    def get_optuna_db_path(cls, model_name):
        short_name = cls.get_model_short_name(model_name)
        return os.path.join(cls.CHECKPOINT_DIR, f"{short_name}_refined.db")

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 60)
        print(f"{'AKTİF KONFIGÜRASYON (Experiment ' + str(cls.EXPERIMENT_ID) + ')':^60}")
        print("=" * 60)
        print(f" Checkpoint Dir    : {cls.CHECKPOINT_DIR}")
        print(f" Results Dir       : {cls.RESULTS_DIR}")
        print(f" Comet Project     : {cls.COMET_PROJECT_NAME_PREFIX}")
        print("-" * 60)
        print(f" Train Data        : {cls.DATA_PATH_TRAIN}")
        print(f" Val Data          : {cls.DATA_PATH_VAL}")
        print(f" Test Data         : {cls.DATA_PATH_TEST}")
        print("-" * 60)
        print(f" Loss Function     : {cls.LOSS_FUNCTION}")
        print(f" Trials / Epochs   : {cls.NUMBER_TRIALS} / {cls.NUMBER_EPOCHS}")
        print("=" * 60 + "\n")


# Dosya ilk yüklendiğinde mevcut ID'ye göre ayarları yap
Config._setup()