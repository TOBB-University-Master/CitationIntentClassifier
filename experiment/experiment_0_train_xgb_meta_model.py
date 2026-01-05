import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import comet_ml
import optuna
import warnings
import functools

# --- 0. Sabitler ve Dosya Yolları ---
CHECKPOINT_DIR = "../_train_meta_104"
COMET_PROJECT_NAME_PREFIX = "experiment-0-meta-models-104"
MODEL_SHORT_NAME = "xgb"

TRAIN_PATH = os.path.join('../data', 'data_v2_train_one_hot_104.csv')
VAL_PATH = os.path.join('../data', 'data_v2_val_one_hot_104.csv')

BEST_MODEL_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, MODEL_SHORT_NAME)
MODEL_PATH = os.path.join(BEST_MODEL_OUTPUT_DIR, 'best_xgb_meta_model.joblib')
ENCODER_PATH = os.path.join(BEST_MODEL_OUTPUT_DIR, 'best_xgb_label_encoder.joblib')
# -------------------------------------------------------------------------

COMET_API_KEY = os.environ.get("COMET_API_KEY", "LrkBSXNSdBGwikgVrzE2m73iw")
COMET_WORKSPACE = os.environ.get("COMET_WORKSPACE", "kemalsami")

N_TRIALS = 50
EARLY_STOPPING_ROUNDS = 10


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

    y_train = train_df['true_label']
    y_val = validation_df['true_label']
    X_train = train_df.drop(columns=['citation_id', 'true_label'])
    X_val = validation_df.drop(columns=['citation_id', 'true_label'])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    print("Veri hazırlama tamamlandı (Scaler kullanılmadı).")

    return (X_train, y_train_encoded, X_val, y_val_encoded, X_train.columns, label_encoder, y_val)


def objective(trial, data):
    """
    Optuna'nın her bir 'deneme' (trial) için çalıştıracağı fonksiyon.
    XGBoost modelini eğitir, değerlendirir ve accuracy skorunu döndürür.
    """
    (X_train, y_train_encoded, X_val, y_val_encoded, feature_names, label_encoder, y_val) = data
    num_classes = len(label_encoder.classes_)

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
        # --- 2. Hiperparametreleri Optuna ile Belirle ---
        xgb_params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 400, 2000),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            'gamma': trial.suggest_float("gamma", 1e-3, 1.0, log=True),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),  # L1
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),  # L2
            # ---!!! DEĞİŞİKLİK BURADA (1/4) !!!---
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS  # Parametre constructor'a eklendi
        }

        if experiment:
            experiment.log_parameters(xgb_params)

        # --- 3. Modeli Eğit ---
        model = xgb.XGBClassifier(**xgb_params)

        # ---!!! DEĞİŞİKLİK BURADA (2/4) !!!---
        # 'early_stopping_rounds' parametresi fit() çağrısından kaldırıldı.
        fit_call = lambda: model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_val, y_val_encoded)],
            verbose=False
            # early_stopping_rounds parametresi buradan kaldırıldı!
        )

        if experiment:
            with experiment.train():
                fit_call()
        else:
            fit_call()
        # --- DEĞİŞİKLİK SONU ---

        y_pred_encoded = model.predict(X_val)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_val, y_pred)

        if experiment:
            with experiment.validate():
                experiment.log_metric("accuracy", accuracy)
                experiment.log_metric("best_iteration", model.best_iteration)

        print(
            f"Trial {trial.number:03d} | Accuracy: {accuracy:.4f} | Best Iter: {model.best_iteration} | Params: {trial.params}")
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

    (X_train, y_train_encoded, X_val, y_val_encoded, feature_names, label_encoder, y_val) = data
    num_classes = len(label_encoder.classes_)

    os.makedirs(BEST_MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"Nihai model ve artifact'lar şu dizine kaydedilecek: {BEST_MODEL_OUTPUT_DIR}")

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
        final_params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            # ---!!! DEĞİŞİKLİK BURADA (3/4) !!!---
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS  # Parametre constructor'a eklendi
        }
        final_params.update(best_params)

        model = xgb.XGBClassifier(**final_params)

        # ---!!! DEĞİŞİKLİK BURADA (4/4) !!!---
        # 'early_stopping_rounds' parametresi fit() çağrısından kaldırıldı.
        model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_val, y_val_encoded)],
            verbose=100
            # early_stopping_rounds parametresi buradan kaldırıldı!
        )
        # --- DEĞİŞİKLİK SONU ---

        print("Nihai model eğitimi tamamlandı.")

        # --- 4. Modeli Değerlendir ve Logla ---
        y_pred_encoded = model.predict(X_val)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_val, y_pred)
        class_report_str = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
        print(f"\nNihai Model Doğruluğu (Accuracy): {accuracy:.4f}")
        print("Sınıflandırma Raporu:")
        print(class_report_str)

        if experiment:
            with experiment.validate():
                experiment.log_metric("accuracy", accuracy)
                experiment.log_metric("best_iteration", model.best_iteration)
                experiment.log_text(f"Classification Report:\n{class_report_str}")
                experiment.log_confusion_matrix(
                    y_val, y_pred, labels=list(label_encoder.classes_)
                )

        # --- 5. Modeli ve Artifact'ları SABİT YOLA Kaydet ---
        print("\nNihai model ve artifact'lar diske kaydediliyor...")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)

        print(f"Model kaydedildi: '{MODEL_PATH}'")
        print(f"Encoder kaydedildi: '{ENCODER_PATH}'")

        if experiment:
            print("Artifact'lar Comet ML'e yükleniyor...")
            experiment.log_model("best_xgb_meta_model", MODEL_PATH)
            experiment.log_model("best_xgb_label_encoder", ENCODER_PATH)

    except Exception as e:
        print(f"HATA: Nihai model eğitimi/kaydı sırasında hata: {e}")
    finally:
        if experiment:
            experiment.end()
            print("Nihai Comet ML deneyi sonlandırıldı.")


def main():
    data = load_and_prep_data()
    if data is None:
        return

    print("\n" + "=" * 50)
    print(f">>> Optuna Optimizasyonu Başlatılıyor (N_TRIALS = {N_TRIALS})")
    print(f"Çalışma Adı: {MODEL_SHORT_NAME}-study")
    print(f"Ana Çıktı Klasörü: {CHECKPOINT_DIR}")
    print("=" * 50 + "\n")

    objective_with_data = functools.partial(objective, data=data)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    storage_path = f"sqlite:///{CHECKPOINT_DIR}/{MODEL_SHORT_NAME}_study.db"

    study = optuna.create_study(
        study_name=f"{MODEL_SHORT_NAME}-study",
        storage=storage_path,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective_with_data, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 50)
    print(">>> Optimizasyon Tamamlandı!")

    best_trial = study.best_trial
    best_trial_number = best_trial.number

    print(f"En iyi deneme (Best trial): {best_trial_number}")
    print(f"En iyi doğruluk (Best accuracy): {best_trial.value:.4f}")
    print("En iyi parametreler (Best parameters):")
    print(study.best_params)
    print("=" * 50 + "\n")

    train_and_log_best_model(study.best_params, data, best_trial_number)

    print("\nTüm süreç tamamlandı!")


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='comet_ml')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message=".*callbacks.*", category=UserWarning)
    main()