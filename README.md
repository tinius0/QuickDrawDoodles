# Quick, Draw! Doodle Classifier

[![Status](https://img.shields.io/badge/status-WIP-yellow)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project Overview
This repository contains a **neural network built from scratch** for classifying doodles from the **Quick, Draw!** dataset. The main goal is to create a model that can recognize hand-drawn images and predict what the user is drawing in real-time.  

> **Work in Progress:** The current model is **not fully complete**, as integrating the Deep Neural Network (DNN) and Convolutional Neural Network (CNN) architectures has been challenging.  

> **Training Limitation:** The model is currently trained on a **custom subset of figures** from the Quick, Draw! dataset, not the full 345 categories.

## Current Status
- DNN + CNN combination is under development  
- Debugging feature extraction and classification pipeline  
- Planning real-time drawing recognition via a web interface  

## Key Features
- **Real-time Drawing Recognition:** Draw doodles on a web interface and get instant predictions  
- **Custom Training:** Scripts for training the neural network on a subset of Quick, Draw! figures  
- **Performance Evaluation:** Evaluate model accuracy, precision, and recall on the test set  

## Technology & Dataset
- **Neural Network:** Built from scratch using Python  
- **Dataset:** Quick, Draw! (subset of categories)  
- **Architecture:** Combination of CNN for feature extraction + DNN for classification  
- **Limitations:** Model is a work in progress and currently trained only on selected figures  

## Roadmap
- [ ] Complete CNN + DNN integration  
- [ ] Expand dataset coverage  
- [ ] Optimize real-time prediction performance  
- [ ] Add visualizations of training progress and predictions  

## Notes
This project is an exploration of **building neural networks from scratch** without using high-level frameworks like TensorFlow or PyTorch. Expect bugs and incomplete functionality as the architecture is refined.

---

Made with ❤️ by tinius0
