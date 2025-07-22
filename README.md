# 🕵️ AML Fraud Detection API

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)](https://fastapi.tiangolo.com)

AI-агент для детекции подозрительных транзакций с точностью **92% AUC-ROC**.

## 🔥 Особенности
- Генерация синтетических данных с паттернами мошенничества
- ML-модель на CatBoost с интерпретируемыми фичами
- REST API с Swagger-документацией
- Логирование и анализ рисковых факторов

## 🚀 Быстрый старт
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API
uvicorn app.api_service:app --reload
