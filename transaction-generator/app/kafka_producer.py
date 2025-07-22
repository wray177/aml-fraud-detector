import json
import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
from faker import Faker
import numpy as np

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Кастомный энкодер для numpy-типов"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class TransactionGenerator:
    def __init__(self):
        self.producer = self._create_producer()
        self.fake = Faker()

    def _create_producer(self):
        try:
            producer = KafkaProducer(
                bootstrap_servers=['localhost:29092'],
                value_serializer=lambda v: json.dumps(v, cls=NumpyEncoder).encode('utf-8'),
                acks='all',
                retries=3,
                linger_ms=10
            )
            logger.info("Successfully connected to Kafka at localhost:29092")
            return producer
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    def _generate_transaction(self):
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])
        tx = {
            "transaction_id": self.fake.uuid4(),
            "timestamp": int(time.time() * 1000),
            "currency": np.random.choice(["USD", "EUR", "RUB"]).item(),
            "sender": f"CUST_{self.fake.uuid4()[:8]}",
            "location": self.fake.country_code(),
            "is_fraud": int(is_fraud)
        }

        if is_fraud:
            fraud_type = np.random.choice(["micro", "split", "round", "recurring", "foreign"])
            
            if fraud_type == "micro":
                tx.update({
                    "amount": round(np.random.uniform(0.1, 5), 2),
                    "recipient": f"SUSP_MICRO_{self.fake.uuid4()[:5]}",
                    "fraud_pattern": "micro"
                })
            elif fraud_type == "split":
                tx.update({
                    "amount": round(np.random.uniform(900, 999), 2),
                    "recipient": f"SUSP_SPLIT_{self.fake.uuid4()[:5]}",
                    "fraud_pattern": "split"
                })
            elif fraud_type == "round":
                tx.update({
                    "amount": float(np.random.choice([100, 500, 1000])),
                    "recipient": f"SUSP_ROUND_{self.fake.uuid4()[:5]}",
                    "fraud_pattern": "round"
                })
            elif fraud_type == "recurring":
                tx.update({
                    "amount": round(np.random.uniform(10, 50), 2),
                    "recipient": f"SUSP_RECUR_{self.fake.uuid4()[:5]}",
                    "fraud_pattern": "recurring"
                })
            else:  # foreign
                tx.update({
                    "amount": round(np.random.uniform(200, 500), 2),
                    "recipient": f"SUSP_FOREIGN_{self.fake.uuid4()[:5]}",
                    "currency": "USD",
                    "fraud_pattern": "foreign"
                })
        else:
            tx.update({
                "amount": round(np.random.lognormal(5, 1.5), 2),
                "recipient": f"CUST_{self.fake.uuid4()[:8]}"
            })

        return tx

    def run(self):
        try:
            while True:
                tx = self._generate_transaction()
                try:
                    future = self.producer.send('transactions', tx)
                    future.get(timeout=10)
                    logger.info(f"Sent transaction: {tx['transaction_id']}, "
                                f"Amount: {tx['amount']}, "
                                f"Fraud: {bool(tx['is_fraud'])}")
                except KafkaError as e:
                    logger.error(f"Failed to send transaction: {e}")
                time.sleep(np.random.uniform(0.5, 1))  # Случайные интервалы
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.producer.close()
            logger.info("Producer closed")

if __name__ == "__main__":
    TransactionGenerator().run()