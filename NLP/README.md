# Sexism Classification via Prompt-Based LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#LICENSE)  
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/sexism-classifier/ci.yml)](#)  

---

## 🚀 Project Overview

Prompt-based classification of sentences as **sexist** or **not sexist**, leveraging three open-source LLMs:

- **Mistral v3**  
- **Llama v3.1**  
- **Phi3-Mini**  

We compare zero-shot, few-shot, and retrieval-augmented few-shot setups, plus ensemble strategies.

---

## 📋 Table of Contents

1. [Features](#features)  
2. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
3. [Usage](#usage)  
   - [Zero-Shot Inference](#zero-shot-inference)  
   - [Few-Shot Inference](#few-shot-inference)  
   - [Retrieval-Augmented Inference](#retrieval-augmented-inference)  
   - [Ensemble Methods](#ensemble-methods)  
4. [Results](#results)  
5. [Code Structure](#code-structure)  
6. [Performance & Latency](#performance--latency)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Contact](#contact)

---

## ✨ Features

- **Zero-Shot & Few-Shot** pipelines with customizable prompts  
- **Intelligent Example Selection** via TF-IDF, Word2Vec, BERT  
- **Ensemble Strategies**: simple voting, weighted voting, logistic-regression meta-ensemble  
- **Prompt Tokenization Optimization** for runtime speedup  
- Clear logging & evaluation on held-out test sets  

---

## 🛠 Getting Started

### Prerequisites

- Python 3.10+  
- `pip` (or `conda`)  
- Access to LLM APIs (e.g. Mistral, Llama, Phi3 credentials)

## 📊 Results

| Model      | Zero-Shot | Few-Shot |
|------------|:---------:|:--------:|
| Llama v3.1 |    67%    |   56%    |
| Phi3-Mini  |    61%    |   61%    |
| Mistral v3 |  **71%**  | **71%**  |






