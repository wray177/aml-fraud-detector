import json
import pandas as pd
from kafka import KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_data(sample_size=1000):
    """Собирает данные из Kafka без зависаний"""
    try:
        consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers='localhost:29092',
            auto_offset_reset='earliest',
            consumer_timeout_ms=5000,  # 5 секунд на ожидание
            api_version=(2, 5, 0)  # Явно указываем версию Kafka
        )
        
        data = []
        for i, msg in enumerate(consumer):
            if i >= sample_size:
                break
            try:
                tx = json.loads(msg.value.decode('utf-8'))
                data.append(tx)
                logger.info(f"Получена транзакция: {tx['transaction_id']}")
            except Exception as e:
                logger.error(f"Ошибка в сообщении: {e}")
        
        consumer.close()
        return pd.DataFrame(data) if data else None

    except Exception as e:
        logger.error(f"Ошибка подключения к Kafka: {e}")
        return None

if __name__ == "__main__":
    logger.info("Начинаем сбор данных...")
    df = collect_data(sample_size=1000)  # Сначала проверим на 100 сообщениях
    
    if df is not None:
        logger.info(f"Успешно собрано {len(df)} транзакций")
        df.to_csv('transactions_cleaned.csv', index=False)
        logger.info("Данные сохранены в transactions_cleaned.csv")
    else:
        logger.error("Не удалось получить данные!")