# -*- coding: utf-8 -*-
"""
Module de chargement des données depuis Azure Blob Storage
"""

import streamlit as st
import pandas as pd
import io
from azure.storage.blob import BlobServiceClient
from config import Config

@st.cache_data(ttl=3600, show_spinner="📥 Chargement des données depuis Azure...")
def load_data_from_azure():
    """
    Charge les données TER depuis Azure Blob Storage
    Cache les données pendant 1 heure
    """
    try:
        # Connexion à Azure
        service_client = BlobServiceClient.from_connection_string(
            Config.AZURE_CONNECTION_STRING
        )
        
        # Récupération du blob
        blob_client = service_client.get_blob_client(
            container=Config.AZURE_CONTAINER_NAME,
            blob=Config.AZURE_BLOB_NAME
        )
        
        # Téléchargement
        stream = blob_client.download_blob().readall()
        
        # Lecture selon le format
        if Config.AZURE_BLOB_NAME.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(stream))
        elif Config.AZURE_BLOB_NAME.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(stream))
        else:
            st.error("❌ Format de fichier non supporté. Utilisez .xlsx ou .csv")
            return None
        
        # Nettoyage de base
        df = clean_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement depuis Azure : {e}")
        return None

def clean_data(df):
    """Nettoie et prépare les données"""
    
    # Conversion des dates
    date_columns = ['date', 'Date', 'DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break
    
    # Standardisation des noms de colonnes (minuscules, sans espaces)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace("'", '')
    
    # Suppression des doublons
    df = df.drop_duplicates()
    
    # Tri par date si disponible
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=False)
    
    return df

def get_data_summary(df):
    """Génère un résumé statistique des données"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'date_range': None,
        'regions': [],
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Plage de dates
    if 'date' in df.columns:
        summary['date_range'] = (df['date'].min(), df['date'].max())
    
    # Régions uniques
    if 'region' in df.columns:
        summary['regions'] = df['region'].unique().tolist()
    
    return summary
