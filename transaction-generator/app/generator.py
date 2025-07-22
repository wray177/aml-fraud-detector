# ~/aml-fraud-detector/transaction-generator/app/kafka_producer.py
import json
import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
from faker import Faker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionGenerator:
    def __init__(self):
        self.producer = self._create_producer()
        self.fake = Faker()

    def _create_producer(self):
        try:
            producer = KafkaProducer(
                bootstrap_servers='localhost:29092',
                value_serializer=lambda v: json.dumps({
                    **v,
                    'amount': float(v['amount']),
                    'is_fraud': int(v['is_fraud'])
                }).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info("Connected to Kafka successfully")
            return producer
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def _generate_transaction(self):
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])
        amount = np.random.lognormal(5, 1.5)
        if is_fraud:
            amount = np.random.uniform(0.1, 10)
        return {
            "transaction_id": self.fake.uuid4(),
            "amount": round(float(amount), 2),
            "currency": str(np.random.choice(["USD", "EUR", "RUB"])),
            "timestamp": int(time.time() * 1000),
            "is_fraud": int(is_fraud)
        }

    def run(self):
        try:
            while True:
                tx = self._generate_transaction()
                try:
                    future = self.producer.send('transactions', tx)
                    future.get(timeout=10)
                    logger.info(f"Sent: {tx}")
                except KafkaError as e:
                    logger.error(f"Send failed: {e}")
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.producer.close()

if __name__ == "__main__":
    TransactionGenerator().run()