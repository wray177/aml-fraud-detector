from kafka import KafkaConsumer
import json
import logging

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_consumer():
    try:
        consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers='localhost:29092',
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=10000  # Таймаут 10 сек
        )
        logger.info("Consumer created successfully")
        return consumer
    except Exception as e:
        logger.error(f"Failed to create consumer: {e}")
        raise

def detect_fraud(transaction):
    """Логика обнаружения подозрительных транзакций"""
    try:
        # Пример простого правила: микроплатежи считаем подозрительными
        if transaction['amount'] < 1.0:
            return True
        return False
    except KeyError as e:
        logger.error(f"Invalid transaction format: {e}")
        return False

def process_messages():
    consumer = create_consumer()
    try:
        logger.info("Starting to consume messages...")
        for message in consumer:
            try:
                tx = message.value
                logger.info(f"Received transaction: {tx}")
                
                if detect_fraud(tx):
                    logger.warning(f"FRAUD DETECTED! Transaction: {tx}")
                else:
                    logger.info("Transaction looks legitimate")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        consumer.close()
        logger.info("Consumer closed")

if __name__ == "__main__":
    process_messages()