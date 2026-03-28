# config.py
# -*- coding: utf-8 -*-
"""
Configuration centralisée pour l'Assistant IA TER
"""

import streamlit as st


class Config:
    """Configuration de l'application"""

    # Azure Blob Storage
    AZURE_CONNECTION_STRING = st.secrets.get("AZURE_STORAGE_CONNECTION_STRING", "")
    AZURE_CONTAINER_NAME = st.secrets.get("AZURE_CONTAINER_NAME", "ztacontainer")
    AZURE_BLOB_NAME = st.secrets.get("AZURE_BLOB_NAME", "ter_ponctualite_2024_2025.xlsx")

    # Mistral AI — mistral-large-latest est recommandé :
    # bien plus puissant que mistral-small pour le raisonnement sur données,
    # et disponible avec le même niveau d'API key gratuite.
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", "")
    MISTRAL_MODEL = "mistral-large-latest"

    # Configuration de l'app
    APP_TITLE = "🚆 Assistant IA - Analyse TER SNCF"
    APP_ICON = "🚆"
    LAYOUT = "wide"

    # Colonnes attendues dans le dataset TER
    EXPECTED_COLUMNS = {
        'date': 'Date',
        'annee': 'Année',
        'mois': 'Mois',
        'region': 'Région',
        'nombre_trains_prevus': 'Trains prévus',
        'nombre_trains_circules': 'Trains circulés',
        'nombre_trains_a_l_heure': 'Trains à l\'heure',
        'nombre_trains_retard': 'Trains en retard',
        'nombre_trains_supprimes': 'Trains supprimés',
        'taux_regularite': 'Taux de régularité (%)'
    }

    # Seuils d'alerte
    SEUIL_REGULARITE_CRITIQUE = 85.0
    SEUIL_REGULARITE_NORMAL = 92.0


def check_config():
    """Vérifie que toutes les configurations critiques sont présentes"""
    missing = []

    if not Config.MISTRAL_API_KEY:
        missing.append("MISTRAL_API_KEY")

    if missing:
        st.error(f"⚠️ Configuration manquante : {', '.join(missing)}")
        st.info("💡 Configurez ces secrets dans les paramètres Streamlit Cloud ou dans .streamlit/secrets.toml")
        st.stop()
