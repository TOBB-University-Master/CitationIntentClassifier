import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import comet_ml
import optuna
import warnings
import functools

# --- 0. Sabitler ve Dosya Yolları ---
CHECKPOINT_DIR = "checkpoints_v0"
COMET_PROJECT_NAME_PREFIX = "experiment-0-meta-models"
MODEL_SHORT_NAME = "lr"

# Veri yolları
TRAIN_PATH = os.path.join('data', 'data_v2_train_one_hot.csv')
VAL_PATH = os.path.join('data', 'data_v2_val_one_hot.csv')
BEST_MODEL_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, MODEL_SHORT_NAME)
MODEL_PATH = os.path.join(BEST_MODEL_OUTPUT_DIR, 'best_lr_meta_model.joblib')
ENCODER_PATH = os.path.join(BEST_MODEL_OUTPUT_DIR, 'best_lr_label_encoder.joblib')
SCALER_PATH = os.path.join(BEST_MODEL_OUTPUT_DIR, 'best_lr_scaler.joblib')

# Comet ML API Bilgileri
COMET_API_KEY = os.environ.get("COMET_API_KEY", "LrkBSXNSdBGwikgVrzE2m73iw")
COMET_WORKSPACE = os.environ.get("COMET_WORKSPACE", "kemalsami")

# Optuna Çalışma Ayarları
N_TRIALS = 50


def load_and_prep_data():
    """Veri yükleme, ayırma ve ölçeklendirme işlemlerini yapar."""
    print("\n>>> Veri yükleniyor ve hazırlanıyor...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        validation_df = pd.read_csv(VAL_PATH)
        print(f"Eğitim seti yüklendi: {train_df.shape[0]} satır")
        print(f"Doğrulama seti yüklendi: {validation_df.shape[0]} satır")
    except FileNotFoundError:
        print(f"\nHATA: '{TRAIN_PATH}' veya '{VAL_PATH}' bulunamadı.")
        return None

    # Veriyi ayır, kodla ve ölçekle...
    y_train = train_df['true_label']
    y_val = validation_df['true_label']
    X_train = train_df.drop(columns=['citation_id', 'true_label'])
    X_val = validation_df.drop(columns=['citation_id', 'true_label'])
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("Veri hazırlama tamamlandı.")

    return (X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded,
            X_train.columns, label_encoder, scaler, y_val)


def objective(trial, data):
    """
    Optuna'nın her bir 'deneme' (trial) için çalıştıracağı fonksiyon.
    Modeli eğitir, değerlendirir ve accuracy skorunu döndürür.
    (Bu fonksiyon artık disk'e hiçbir model kaydetmez)
    """

    (X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded,
     feature_names, label_encoder, scaler, y_val) = data

    # --- 1. Comet ML Deneyini Başlat ---
    experiment = None
    try:
        experiment = comet_ml.Experiment(
            api_key=COMET_API_KEY,
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{MODEL_SHORT_NAME}-study",
            workspace=COMET_WORKSPACE,
        )
        experiment.set_name(f"trial_{trial.number:03d}")
        experiment.add_tag(MODEL_SHORT_NAME)
    except Exception as e:
        print(f"HATA: Comet ML başlatılamadı (Trial {trial.number}). {e}")

    try:
        # --- 2. Hiperparametreleri Belirle ---
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        if penalty == 'l1':
            solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
        else:
            solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "saga"])

        lr_params = {
            'C': trial.suggest_float("C", 1e-4, 1e2, log=True),
            'penalty': penalty,
            'solver': solver,
            'class_weight': trial.suggest_categorical("class_weight", [None, "balanced"]),
            'max_iter': trial.suggest_categorical("max_iter", [1000, 2000, 5000]),
            'random_state': 42
        }

        if experiment:
            experiment.log_parameters(lr_params)

        # --- 3. Modeli Eğit ---
        base_lr = LogisticRegression(**lr_params)
        model = OneVsRestClassifier(base_lr)

        if experiment:
            with experiment.train():
                model.fit(X_train_scaled, y_train_encoded)
        else:
            model.fit(X_train_scaled, y_train_encoded)

        # --- 4. Modeli Değerlendir ---
        y_pred_encoded = model.predict(X_val_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_val, y_pred)

        if experiment:
            with experiment.validate():
                experiment.log_metric("accuracy", accuracy)

        print(f"Trial {trial.number:03d} | Accuracy: {accuracy:.4f} | Params: {trial.params}")
        return accuracy

    except Exception as e:
        print(f"HATA: Trial {trial.number} başarısız oldu. Hata: {e}")
        return 0.0
    finally:
        if experiment:
            experiment.end()


def train_and_log_best_model(best_params, data, best_trial_number):
    """Optuna'dan gelen en iyi parametrelerle nihai modeli eğitir ve loglar."""

    print("\n" + "=" * 50)
    print(">>> En İyi Parametrelerle Nihai Model Eğitiliyor...")
    print(f"Kaynak Deneme (Source Trial): {best_trial_number}")
    print(f"Parametreler: {best_params}")
    print("=" * 50 + "\n")

    (X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, feature_names, label_encoder, scaler, y_val) = data

    os.makedirs(BEST_MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"Nihai model ve artifact'lar şu dizine kaydedilecek: {BEST_MODEL_OUTPUT_DIR}")


    # --- 2. Nihai Comet ML Deneyini Başlat ---
    experiment = None
    try:
        experiment = comet_ml.Experiment(
            api_key=COMET_API_KEY,
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{MODEL_SHORT_NAME}-study",
            workspace=COMET_WORKSPACE,
        )

        experiment.set_name(f"FINAL-BEST-MODEL (from trial {best_trial_number})")
        experiment.log_parameters(best_params)
        experiment.log_parameter("source_trial_number", best_trial_number)

    except Exception as e:
        print(f"HATA: Nihai Comet ML deneyi başlatılamadı. {e}")

    try:
        # --- 3. Modeli Eğit ---
        final_params = best_params.copy()
        if 'solver_l1' in final_params:
            final_params['solver'] = final_params.pop('solver_l1')
        if 'solver_l2' in final_params:
            final_params['solver'] = final_params.pop('solver_l2')
        final_params['random_state'] = 42

        base_lr = LogisticRegression(**final_params)
        model = OneVsRestClassifier(base_lr)

        model.fit(X_train_scaled, y_train_encoded)
        print("Nihai model eğitimi tamamlandı.")

        # --- 4. Modeli Değerlendir ve Logla ---
        y_pred_encoded = model.predict(X_val_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_val, y_pred)
        class_report_str = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
        print(f"\nNihai Model Doğruluğu (Accuracy): {accuracy:.4f}")

        if experiment:
            with experiment.validate():
                experiment.log_metric("accuracy", accuracy)
                experiment.log_text(f"Classification Report:\n{class_report_str}")
                experiment.log_confusion_matrix(
                    y_val, y_pred, labels=list(label_encoder.classes_)
                )

        # --- 5. Modeli ve Artifact'ları SABİT YOLA Kaydet ---
        print("\nNihai model ve artifact'lar diske kaydediliyor...")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        print(f"Model kaydedildi: '{MODEL_PATH}'")
        print(f"Encoder kaydedildi: '{ENCODER_PATH}'")
        print(f"Scaler kaydedildi: '{SCALER_PATH}'")

        if experiment:
            print("Artifact'lar Comet ML'e yükleniyor...")
            experiment.log_model("best_lr_meta_model", MODEL_PATH)
            experiment.log_model("best_lr_label_encoder", ENCODER_PATH)
            experiment.log_model("best_lr_scaler", SCALER_PATH)

    except Exception as e:
        print(f"HATA: Nihai model eğitimi/kaydı sırasında hata: {e}")
    finally:
        if experiment:
            experiment.end()
            print("Nihai Comet ML deneyi sonlandırıldı.")


def main():
    # 1. Veriyi yükle ve hazırla
    data = load_and_prep_data()
    if data is None:
        return

    # 2. Optuna Çalışmasını (Study) Başlat
    print("\n" + "=" * 50)
    print(f">>> Optuna Optimizasyonu Başlatılıyor (N_TRIALS = {N_TRIALS})")
    print(f"Çalışma Adı: {MODEL_SHORT_NAME}-study")
    print(f"Ana Çıktı Klasörü: {CHECKPOINT_DIR}")
    print("=" * 50 + "\n")

    objective_with_data = functools.partial(objective, data=data)

    # Optuna DB yolunu CHECKPOINT_DIR içine al
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    storage_path = f"sqlite:///{CHECKPOINT_DIR}/{MODEL_SHORT_NAME}_study.db"

    study = optuna.create_study(
        study_name=f"{MODEL_SHORT_NAME}-study",
        storage=storage_path,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective_with_data, n_trials=N_TRIALS, show_progress_bar=True)

    # 3. Optimizasyon Sonuçlarını Göster
    print("\n" + "=" * 50)
    print(">>> Optimizasyon Tamamlandı!")

    best_trial = study.best_trial
    best_trial_number = best_trial.number  # Bu bilgiyi Comet'e loglamak için alıyoruz

    print(f"En iyi deneme (Best trial): {best_trial_number}")
    print(f"En iyi doğruluk (Best accuracy): {best_trial.value:.4f}")
    print("En iyi parametreler (Best parameters):")
    print(study.best_params)
    print("=" * 50 + "\n")

    # 4. En iyi parametrelerle ve 'best_trial_number' (loglama için) ile nihai modeli eğit
    train_and_log_best_model(study.best_params, data, best_trial_number)

    print("\nTüm süreç tamamlandı!")


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='comet_ml')
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()