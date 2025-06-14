# Sarcasm Detection with Neural Networks

This project implements a sarcasm detection model using TensorFlow and Keras. The model is trained on the [Sarcasm Headlines Dataset v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), which contains news headlines labeled as sarcastic or not.

## Features

- Loads and preprocesses the dataset from JSON.
- Tokenizes and pads the text data.
- Builds and trains a neural network for binary classification.
- Evaluates model performance and allows for prediction on custom sentences.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

Install dependencies with:

```sh
pip install tensorflow numpy
```

## Usage

1. **Download the Dataset**

   Place `Sarcasm_Headlines_Dataset_v2.json` in a `data/` directory relative to the notebook.

2. **Run the Notebook**

   Open [`sarcasm-detection (1).ipynb`](c:/Users/sanje/Downloads/sarcasm-detection%20(1).ipynb) in VS Code or Jupyter and run all cells.

3. **Training**

   The notebook will:
   - Load and preprocess the data.
   - Split into training and testing sets.
   - Train a neural network model.
   - Output accuracy and loss metrics.

4. **Prediction**

   You can test the model on your own sentences by modifying the following cell:

   ````python
   # Predict sarcasm for a custom sentence
   sent1 = ["Your custom headline here"]
   seq1 = tokenizer.texts_to_sequences(sent1)
   pad1 = pad_sequences(seq1, maxlen=max_lenght, padding=padding_type, truncating=trunc_type)
   prediction = model.predict(pad1)[0][0]
   print(prediction)
