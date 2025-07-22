import pandas as pd
import numpy as np
from datetime import datetime

# Маппинг городов/регионов на коды стран
LOCATION_MAPPING = {
    # США
    'NY': 'US', 'LA': 'US', 'SF': 'US', 'TX': 'US', 'FL': 'US',
    # Россия
    'MSK': 'RU', 'SPB': 'RU', 'NSK': 'RU', 'EKB': 'RU',
    # Другие страны
    'LON': 'GB', 'PAR': 'FR', 'BER': 'DE', 'TOK': 'JP', 'DEL': 'IN'
}

def map_location_to_country(location: str) -> str:
    """Конвертирует 'NY' → 'US', 'Moscow' → 'RU' и т.д."""
    if location in LOCATION_MAPPING:
        return LOCATION_MAPPING[location]
    return location.upper() if len(location) == 2 else None

# Полный список всех фичей, которые ожидает модель
REQUIRED_FEATURES = [
    'amount', 'log_amount', 'is_micro', 'is_small', 'is_round', 'is_night', 
    'is_weekend', 'is_rub', 'is_usd', 'is_eur', 'is_suspicious_recipient',
    'sender_count', 'recipient_count', 'is_recurring', 'is_split'
] + [
    f'loc_{code}' for code in [
        'AD', 'AE', 'AF', 'AG', 'AL', 'AM', 'AO', 'AR', 'AT', 'AU', 'AZ', 'BA', 'BB', 'BD', 'BE', 
        'BF', 'BG', 'BH', 'BI', 'BJ', 'BN', 'BO', 'BR', 'BS', 'BT', 'BW', 'BY', 'BZ', 'CA', 'CD', 
        'CF', 'CG', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CV', 'CY', 'CZ', 'DE', 'DJ', 
        'DK', 'DM', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FM', 'FR', 'GA', 
        'GB', 'GD', 'GE', 'GH', 'GM', 'GN', 'GQ', 'GR', 'GT', 'GW', 'GY', 'HN', 'HR', 'HT', 'HU', 
        'ID', 'IE', 'IL', 'IN', 'IQ', 'IR', 'IS', 'IT', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 
        'KM', 'KN', 'KP', 'KR', 'KW', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS', 'LT', 'LU', 'LV', 
        'LY', 'MA', 'MC', 'MD', 'ME', 'MG', 'MH', 'MK', 'ML', 'MM', 'MN', 'MR', 'MT', 'MU', 'MV', 
        'MW', 'MX', 'MY', 'MZ', 'NE', 'NG', 'NI', 'NL', 'NO', 'NP', 'NR', 'NZ', 'OM', 'PA', 'PE', 
        'PG', 'PH', 'PK', 'PL', 'PS', 'PT', 'PW', 'PY', 'QA', 'RO', 'RS', 'RU', 'RW', 'SA', 'SB', 
        'SC', 'SD', 'SE', 'SG', 'SI', 'SK', 'SL', 'SM', 'SN', 'SO', 'SR', 'ST', 'SV', 'SY', 'SZ', 
        'TD', 'TG', 'TH', 'TJ', 'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TV', 'TW', 'TZ', 'UA', 'UG', 
        'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VN', 'VU', 'WS', 'YE', 'ZA', 'ZM', 'ZW'
    ]
]

def prepare_features(df):
    """
    Генерация фичей для AML-детектора (работает как для batch, так и для единичных транзакций)
    
    Args:
        df: DataFrame или dict с полями:
            - amount
            - currency
            - timestamp (unix ms)
            - recipient
            - sender (опционально)
            - location (код страны из списка выше)
    
    Returns:
        DataFrame с дополнительными фичами
    """
    df = df.copy()
    
    # 1. Инициализируем все фичи нулями
    for feature in REQUIRED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    
    # 2. Обрабатываем location (если передан)
    if 'location' in df.columns:
        # Приводим location к стандартному формату
        mapped_loc = map_location_to_country(df['location'].iloc[0])
        if mapped_loc:
            loc_col = f'loc_{mapped_loc}'
            if loc_col in REQUIRED_FEATURES:
                df[loc_col] = 1
            df['location'] = mapped_loc  # Обновляем значение location
    
    # 3. Основные фичи
    df['is_micro'] = (df['amount'] < 10).astype(int)
    df['is_small'] = ((df['amount'] >= 10) & (df['amount'] < 100)).astype(int)
    df['is_round'] = (df['amount'] % 100 == 0).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    
    # 4. Временные фичи
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = df['timestamp'].dt.hour
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 5. Фичи по валютам
    if 'currency' in df.columns:
        df['is_rub'] = (df['currency'] == 'RUB').astype(int)
        df['is_usd'] = (df['currency'] == 'USD').astype(int)
        df['is_eur'] = (df['currency'] == 'EUR').astype(int)
    
    # 6. Фичи по получателям
    if 'recipient' in df.columns:
        df['is_suspicious_recipient'] = df['recipient'].str.startswith(
            ('SUSP_', 'FRAUD_'), na=False).astype(int)
    
    return df

def get_feature_names():
    """Возвращает список фичей, которые генерирует функция prepare_features"""
    return REQUIRED_FEATURES

# Добавляем функцию в список экспортируемых
__all__ = ['prepare_features', 'get_feature_names', 'map_location_to_country']

if __name__ == "__main__":
    # Пример использования для обучения
    df = pd.read_csv('transactions_cleaned.csv')
    df = prepare_features(df)
    
    # Сохраняем только нужные фичи
    df[REQUIRED_FEATURES + ['is_fraud']].to_csv('training_data.csv', index=False)
    print("Файл с фичами сохранён: training_data.csv")