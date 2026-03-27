# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA conversationnel pour l'analyse des données TER
Capacité à générer des graphiques et des analyses textuelles
"""

from mistralai import Mistral
import pandas as pd
import json
from typing import Dict, Any, Optional, Tuple
from config import Config


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER avec génération de graphiques"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'agent avec les données TER
        
        Args:
            df: DataFrame contenant les données TER
        """
        self.df = df
        self.client = Mistral(api_key=Config.MISTRAL_API_KEY)
        self.model = Config.MISTRAL_MODEL
        self.conversation_history = []
        
        # Préparer le contexte des données
        self.data_context = self._prepare_data_context()
        
        print("✅ Agent IA initialisé avec succès")
    
    def _prepare_data_context(self) -> str:
        """Prépare un résumé du contexte des données"""
        
        context_parts = [
            f"📊 Dataset TER : {len(self.df):,} enregistrements",
            f"📋 Colonnes : {', '.join(self.df.columns.tolist())}"
        ]
        
        # Ajouter des statistiques selon les colonnes disponibles
        if 'taux_regularite' in self.df.columns:
            avg_reg = self.df['taux_regularite'].mean()
            min_reg = self.df['taux_regularite'].min()
            max_reg = self.df['taux_regularite'].max()
            context_parts.append(f"📈 Régularité : moyenne={avg_reg:.2f}%, min={min_reg:.2f}%, max={max_reg:.2f}%")
        
        if 'region' in self.df.columns:
            nb_regions = self.df['region'].nunique()
            regions = self.df['region'].unique().tolist()[:10]
            context_parts.append(f"🗺️ Régions : {nb_regions} régions ({', '.join(regions)}...)")
        
        if 'date' in self.df.columns:
            date_min = self.df['date'].min()
            date_max = self.df['date'].max()
            context_parts.append(f"📅 Période : du {date_min} au {date_max}")
        
        # Vérifier si données météo présentes
        weather_cols = [col for col in self.df.columns if 'temperature' in col.lower() or 'precipitation' in col.lower() or 'neige' in col.lower()]
        if weather_cols:
            context_parts.append(f"🌦️ Données météo disponibles : {', '.join(weather_cols[:5])}")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Crée le prompt système pour l'agent"""
        
        return f"""Tu es un assistant IA expert en analyse de données ferroviaires TER (Trains Express Régionaux) en France.

**Contexte des données disponibles :**
{self.data_context}

**Tes capacités :**
1. ✅ Analyser les données et
