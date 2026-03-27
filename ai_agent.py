# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA conversationnel pour l'analyse des données TER
"""

from langchain_mistralai import Mistral
import pandas as pd
from typing import Tuple, Optional
from config import Config


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER"""
    
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
        
        # Ajouter des statistiques
        if 'taux_regularite' in self.df.columns:
            avg_reg = self.df['taux_regularite'].mean()
            context_parts.append(f"📈 Régularité moyenne : {avg_reg:.2f}%")
        
        if 'region' in self.df.columns:
            nb_regions = self.df['region'].nunique()
            context_parts.append(f"🗺️ Nombre de régions : {nb_regions}")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Crée le prompt système pour l'agent"""
        
        return f"""Tu es un assistant IA expert en analyse de données ferroviaires TER en France.

**Contexte des données disponibles :**
{self.data_context}

**Tes capacités :**
- Analyser les données et répondre à des questions
- Calculer des statistiques
- Faire des comparaisons entre régions
- Analyser les tendances temporelles

**Instructions :**
- Réponds en français de manière claire et concise
- Utilise des émojis pour rendre la réponse agréable
- Fournis des chiffres précis quand c'est possible
- Si tu ne peux pas répondre, dis-le clairement

**Colonnes disponibles :**
{', '.join(self.df.columns.tolist())}

Réponds de façon professionnelle et précise !"""
    
    def ask(self, question: str) -> str:
        """
        Pose une question à l'agent
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Réponse de l'agent
        """
        print(f"\n🤔 Question reçue : {question}")
        
        # Ajouter la question à l'historique
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        # Créer les messages pour l'API
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt()
            }
        ] + self.conversation_history
        
        try:
            # Appeler l'API Mistral
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extraire la réponse
            assistant_response = response.choices[0].message.content
            
            # Ajouter la réponse à l'historique
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            print(f"✅ Réponse générée")
            
            return assistant_response
        
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'appel à l'API : {str(e)}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Réinitialise l'historique de conversation"""
        self.conversation_history = []
        print("🔄 Historique de conversation réinitialisé")
    
    def get_conversation_length(self) -> int:
        """Retourne le nombre de messages dans l'historique"""
        return len(self.conversation_history)
