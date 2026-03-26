{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMGU+aL3cGA33D8ECGFaAYZ",
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
        "<a href=\"https://colab.research.google.com/github/ZoubirCHATTI/ai_assistant/blob/main/config.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "soO5V0YgnKoA"
      },
      "outputs": [],
      "source": [
        "# -*- coding: utf-8 -*-\n",
        "\"\"\"\n",
        "Configuration centralisée pour l'Assistant IA TER\n",
        "\"\"\"\n",
        "\n",
        "import streamlit as st\n",
        "\n",
        "class Config:\n",
        "    \"\"\"Configuration de l'application\"\"\"\n",
        "\n",
        "    # Azure Storage\n",
        "    AZURE_CONNECTION_STRING = st.secrets.get(\"AZURE_STORAGE_CONNECTION_STRING\", \"\")\n",
        "    AZURE_CONTAINER_NAME = st.secrets.get(\"AZURE_CONTAINER_NAME\", \"ztacontainer\")\n",
        "    AZURE_BLOB_NAME = st.secrets.get(\"AZURE_BLOB_NAME\", \"ter_ponctualite_2024_2025.xlsx\")\n",
        "\n",
        "    # Mistral AI\n",
        "    MISTRAL_API_KEY = st.secrets.get(\"MISTRAL_API_KEY\", \"\")\n",
        "    MISTRAL_MODEL = \"mistral-small-latest\"\n",
        "\n",
        "    # Configuration de l'app\n",
        "    APP_TITLE = \"🚆 Assistant IA - Analyse TER SNCF\"\n",
        "    APP_ICON = \"🚆\"\n",
        "    LAYOUT = \"wide\"\n",
        "\n",
        "    # Colonnes attendues dans le dataset TER (à adapter selon tes vraies colonnes)\n",
        "    EXPECTED_COLUMNS = {\n",
        "        'date': 'Date',\n",
        "        'annee': 'Année',\n",
        "        'mois': 'Mois',\n",
        "        'region': 'Région',\n",
        "        'nombre_trains_prevus': 'Trains prévus',\n",
        "        'nombre_trains_circules': 'Trains circulés',\n",
        "        'nombre_trains_a_l_heure': 'Trains à l\\'heure',\n",
        "        'nombre_trains_retard': 'Trains en retard',\n",
        "        'nombre_trains_supprimes': 'Trains supprimés',\n",
        "        'taux_regularite': 'Taux de régularité (%)'\n",
        "    }\n",
        "\n",
        "    # Seuils d'alerte\n",
        "    SEUIL_REGULARITE_CRITIQUE = 85.0  # % en dessous duquel c'est critique\n",
        "    SEUIL_REGULARITE_NORMAL = 92.0    # % au-dessus duquel c'est bon\n",
        "\n",
        "# Vérification des secrets au démarrage\n",
        "def check_config():\n",
        "    \"\"\"Vérifie que toutes les configurations sont présentes\"\"\"\n",
        "    missing = []\n",
        "\n",
        "    if not Config.AZURE_CONNECTION_STRING:\n",
        "        missing.append(\"AZURE_STORAGE_CONNECTION_STRING\")\n",
        "    if not Config.MISTRAL_API_KEY:\n",
        "        missing.append(\"MISTRAL_API_KEY\")\n",
        "\n",
        "    if missing:\n",
        "        st.error(f\"⚠️ Configuration manquante : {', '.join(missing)}\")\n",
        "        st.info(\"💡 Configurez ces secrets dans les paramètres Streamlit Cloud ou dans .streamlit/secrets.toml\")\n",
        "        st.stop()"
      ]
    }
  ]
}