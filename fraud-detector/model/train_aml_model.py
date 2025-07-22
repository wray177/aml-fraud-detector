import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, auc
import joblib
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aml_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AML Trainer')

def load_data(filepath):
    """Загрузка и проверка данных"""
    logger.info(f"Загрузка данных из {filepath}")
    df = pd.read_csv(filepath)
    
    # Проверка обязательных колонок
    required_cols = ['transaction_id', 'amount', 'is_fraud']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
    
    return df

def create_features(df):
    """Создание признаков для AML-модели"""
    logger.info("Создание признаков...")
    
    # 1. Фичи из временной метки
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = df['timestamp'].dt.hour
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2. Фичи по суммам
    df['is_micro'] = (df['amount'] < 10).astype(int)
    df['is_small'] = ((df['amount'] >= 10) & (df['amount'] < 100)).astype(int)
    df['is_round'] = (df['amount'] % 100 == 0).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    
    # 3. Фичи по валютам
    if 'currency' in df.columns:
        df['is_rub'] = (df['currency'] == 'RUB').astype(int)
        df['is_usd'] = (df['currency'] == 'USD').astype(int)
        df['is_eur'] = (df['currency'] == 'EUR').astype(int)
    
    # 4. Фичи по отправителям/получателям
    if 'sender' in df.columns:
        df['sender_count'] = df.groupby('sender')['transaction_id'].transform('count')
    
    if 'recipient' in df.columns:
        # Помечаем подозрительных получателей
        suspicious_mask = df['recipient'].str.startswith(('SUSP_', 'FRAUD_'), na=False)
        df['is_suspicious_recipient'] = suspicious_mask.astype(int)
        
        # Считаем популярность получателя
        df['recipient_count'] = df.groupby('recipient')['transaction_id'].transform('count')
    
    # 5. Фичи по паттернам мошенничества
    if 'fraud_pattern' in df.columns:
        df['is_recurring'] = df['fraud_pattern'].str.contains('recurring', na=False).astype(int)
        df['is_split'] = df['fraud_pattern'].str.contains('split', na=False).astype(int)
    
    return df

def select_features(df):
    """Выбор финального набора признаков"""
    base_features = [
        'amount',
        'log_amount',
        'is_micro',
        'is_small',
        'is_round',
        'is_night',
        'is_weekend',
        'is_rub',
        'is_usd',
        'is_eur',
        'is_suspicious_recipient',
        'sender_count',
        'recipient_count',
        'is_recurring',
        'is_split'
    ]
    
    # Выбираем только существующие фичи
    features = [f for f in base_features if f in df.columns]
    
    # Добавляем фичи по местоположению, если есть
    if 'location' in df.columns:
        location_dummies = pd.get_dummies(df['location'], prefix='loc')
        df = pd.concat([df, location_dummies], axis=1)
        features += location_dummies.columns.tolist()
    
    return df, features

def train_model(X, y):
    """Обучение CatBoost модели"""
    logger.info("Начало обучения модели...")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )
    
    # Параметры модели
    scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if sum(y_train) > 0 else 1
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='F1',
        random_seed=42,
        scale_pos_weight=scale_pos_weight,
        verbose=100,
        cat_features=[col for col in X.columns if X[col].dtype == 'object']
    )
    
    # Обучение
    model.fit(
        X_train, 
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    return model, X_test, y_test, scale_pos_weight

def evaluate_model(model, X_test, y_test):
    """Оценка качества модели"""
    logger.info("Оценка модели...")
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Отчёт классификации
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Важность признаков
    plot_feature_importance(model, X_test.columns)

def plot_feature_importance(model, feature_names, top_n=10):
    """Профессиональная визуализация важности признаков для AML"""
    importance = model.feature_importances_
    
    # Создаем DataFrame для удобной фильтрации
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Удаляем нулевые и низковажные фичи
    feat_imp = feat_imp[feat_imp['importance'] > 0.01]  # Показываем только значимые
    
    # Сортируем и берем топ-N
    feat_imp = feat_imp.sort_values('importance', ascending=True).tail(top_n)
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        feat_imp['feature'], 
        feat_imp['importance'],
        color='#3498db',
        height=0.7
    )
    
    # Настройки стиля
    plt.title('Top Fraud Detection Features', fontsize=14, pad=20)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Добавляем значения
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.005, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}',
            va='center',
            fontsize=10
        )
    
    # Подпись для AML-контекста
    plt.text(
        0.5, -0.15, 
        "Most predictive features for transaction fraud detection",
        ha='center', 
        transform=plt.gca().transAxes,
        fontsize=10,
        color='gray'
    )
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Финальная версия графика сохранена")

def save_artifacts(model, features, scale_pos_weight):
    """Сохранение артефактов модели"""
    logger.info("Сохранение артефактов...")
    
    try:
        # Модель
        joblib.dump(model, 'aml_model.joblib')
        
        # Метаданные
        metadata = {
            'model_type': 'CatBoostClassifier',
            'training_date': datetime.now().isoformat(),
            'features': features,
            'metrics': {
                'eval_metric': 'F1',
                'scale_pos_weight': scale_pos_weight
            },
            'model_params': model.get_params()
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Артефакты модели сохранены")
    except Exception as e:
        logger.error(f"Ошибка при сохранении артефактов: {str(e)}")
        raise

def main():
    try:
        # 1. Загрузка данных
        df = load_data('../app/transactions_cleaned.csv')
        
        # 2. Создание признаков
        df = create_features(df)
        
        # 3. Выбор финальных признаков
        df, features = select_features(df)
        X = df[features]
        y = df['is_fraud']
        
        # 4. Обучение модели
        model, X_test, y_test, scale_pos_weight = train_model(X, y)
        
        # 5. Оценка
        evaluate_model(model, X_test, y_test)
        
        # 6. Сохранение
        save_artifacts(model, features, scale_pos_weight)
        
        logger.info("Обучение завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()