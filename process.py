{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMOSKl3A590HmabkXH6W/sF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/andhinisetyakholili/Development-NLP/blob/main/process.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x3XaB3VDCQ12"
      },
      "outputs": [],
      "source": [
        "import json\n",
        "import random\n",
        "import nltk\n",
        "import string\n",
        "import numpy as np\n",
        "import pickle\n",
        "import tensorflow as tf\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "global responses, lemmatizer, tokenizer, le, model, input_shape\n",
        "input_shape = 10\n",
        "\n",
        "# import dataset answer\n",
        "def load_response():\n",
        "    global responses\n",
        "    responses = {}\n",
        "    with open('dataset/Intent_KM.json') as content:\n",
        "        data = json.load(content)\n",
        "    for intent in data['intents']:\n",
        "        responses[intent['tag']]=intent['responses']\n",
        "\n",
        "# import model dan download nltk file\n",
        "def preparation():\n",
        "    load_response()\n",
        "    global lemmatizer, tokenizer, le, model\n",
        "    tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))\n",
        "    le = pickle.load(open('model/labelencoder.pkl', 'rb'))\n",
        "    model = keras.models.load_model('model/chat_model.h5')\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "    nltk.download('punkt', quiet=True)\n",
        "    nltk.download('wordnet', quiet=True)\n",
        "    nltk.download('omw-1.4', quiet=True)\n",
        "\n",
        "# hapus tanda baca\n",
        "def remove_punctuation(text):\n",
        "    texts_p = []\n",
        "    text = [letters.lower() for letters in text if letters not in string.punctuation]\n",
        "    text = ''.join(text)\n",
        "    texts_p.append(text)\n",
        "    return texts_p\n",
        "\n",
        "# mengubah text menjadi vector\n",
        "def vectorization(texts_p):\n",
        "    vector = tokenizer.texts_to_sequences(texts_p)\n",
        "    vector = np.array(vector).reshape(-1)\n",
        "    vector = pad_sequences([vector], input_shape)\n",
        "    return vector\n",
        "\n",
        "# klasifikasi pertanyaan user\n",
        "def predict(vector):\n",
        "    output = model.predict(vector)\n",
        "    output = output.argmax()\n",
        "    response_tag = le.inverse_transform([output])[0]\n",
        "    return response_tag\n",
        "\n",
        "# menghasilkan jawaban berdasarkan pertanyaan user\n",
        "def generate_response(text):\n",
        "    texts_p = remove_punctuation(text)\n",
        "    vector = vectorization(texts_p)\n",
        "    response_tag = predict(vector)\n",
        "    answer = random.choice(responses[response_tag])\n",
        "    return answer"
      ]
    }
  ]
}