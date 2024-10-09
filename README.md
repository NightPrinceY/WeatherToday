# Kaggle Weather Prediction Project 🌤️

## Table of Contents 📚

1. [Introduction](#introduction) 📜
2. [Importing Libraries](#importing-libraries) 📚
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda) 📊
4. [Data Preprocessing](#data-preprocessing) 🔧
5. [Model Architecture](#model-architecture) 🏗️
6. [Model Training](#model-training) 🎓
7. [Model Evaluation](#model-evaluation) 📈
8. [Prediction](#prediction) 🔮
9. [Conclusion](#conclusion) 🏁

---

## Introduction 📜

Welcome to the Weather Prediction Project! In this notebook, we will build a **time series forecasting model** to predict daily minimum temperatures using a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** layers. Our goal is to understand temperature trends and enhance predictive modeling skills.

### Objectives 🎯

- Analyze daily minimum temperature data.
- Build and train a hybrid CNN-LSTM model for temperature prediction.
- Evaluate the model's performance using various metrics.

![Temperature Prediction](image.png)

---

## Importing Libraries 📚

Let's start by importing the necessary libraries. These will help us manage data, create visualizations, and build our model.

```python
import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from absl import logging

logging.set_verbosity(logging.ERROR)
