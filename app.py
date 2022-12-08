{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOW6sXgv8D2i5vRyKqgy7XQ",
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
        "<a href=\"https://colab.research.google.com/github/andhinisetyakholili/Development-NLP/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "k-WNuhJDBcB3"
      },
      "outputs": [],
      "source": [
        "from process import preparation, generate_response\n",
        "from flask import Flask, render_template, request\n",
        "\n",
        "# download nltk\n",
        "preparation()\n",
        "\n",
        "#Start Chatbot\n",
        "app = Flask(__name__)\n",
        "\n",
        "@app.route(\"/\")\n",
        "def home():\n",
        "    return render_template(\"index.html\")\n",
        "\n",
        "@app.route(\"/get\")\n",
        "def get_bot_response():\n",
        "    user_input = str(request.args.get('msg'))\n",
        "    result = generate_response(user_input)\n",
        "    return result\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    app.run(debug=True)"
      ]
    }
  ]
}