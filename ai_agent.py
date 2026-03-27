# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA conversationnel pour l'analyse des données TER
Utilise Groq API avec accès direct aux données
"""

import pandas as pd
import json
from typing import Optional, Dict, Any
import requests
from config import Config


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER avec accès direct aux données"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'agent avec les données TER
        
        Args:
            df: DataFrame contenant les données TER
        """
        self.df = df
        self.api_key = Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.conversation_history = []
        
        # Préparer le contexte des données
        self.data_context = self._prepare_data_context()
        
        print("✅ Agent IA initialisé avec succès")
    
    def _prepare_data_context(self) -> str:
        """Prépare un résumé du contexte des données"""
        
        context_parts = [
            f"📊 Dataset TER : {len(self.df):,} enregistrements",
            f"📋 Colonnes disponibles : {', '.join(self.df.columns.tolist())}"
        ]
        
        # Statistiques globales
        if 'taux_regularite' in self.df.columns:
            avg_reg = self.df['taux_regularite'].mean()
            min_reg = self.df['taux_regularite'].min()
            max_reg = self.df['taux_regularite'].max()
            context_parts.append(f"📈 Régularité : moyenne={avg_reg:.2f}%, min={min_reg:.2f}%, max={max_reg:.2f}%")
        
        if 'region' in self.df.columns:
            nb_regions = self.df['region'].nunique()
            regions = sorted(self.df['region'].unique().tolist())[:10]
            context_parts.append(f"🗺️ Régions ({nb_regions} au total) : {', '.join(regions)}...")
        
        if 'date' in self.df.columns:
            date_min = self.df['date'].min()
            date_max = self.df['date'].max()
            context_parts.append(f"📅 Période : du {date_min} au {date_max}")
        
        return "\n".join(context_parts)
    
    def _analyze_data_for_question(self, question: str) -> str:
        """
        Analyse directement les données pour répondre à la question
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Résultats de l'analyse sous forme de texte
        """
        question_lower = question.lower()
        results = []
        
        try:
            # ═══════════════════════════════════════════════════════════
            # FILTRAGE PAR DATE
            # ═══════════════════════════════════════════════════════════
            
            df_filtered = self.df.copy()
            
            # Détecter le mois
            mois_mapping = {
                'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4,
                'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8, 'aout': 8,
                'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12, 'decembre': 12
            }
            
            month = None
            year = None
            
            for mois_nom, mois_num in mois_mapping.items():
                if mois_nom in question_lower:
                    month = mois_num
                    break
            
            # Détecter l'année
            import re
            year_match = re.search(r'\b(20\d{2})\b', question)
            if year_match:
                year = int(year_match.group(1))
            
            # Appliquer le filtre de date
            if 'date' in df_filtered.columns and (month or year):
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
                if year:
                    df_filtered = df_filtered[df_filtered['date'].dt.year == year]
                    results.append(f"📅 Filtré sur l'année {year}")
                
                if month:
                    df_filtered = df_filtered[df_filtered['date'].dt.month == month]
                    mois_nom_str = list(mois_mapping.keys())[list(mois_mapping.values()).index(month)]
                    results.append(f"📅 Filtré sur le mois de {mois_nom_str}")
            
            # ═══════════════════════════════════════════════════════════
            # FILTRAGE PAR RÉGION
            # ═══════════════════════════════════════════════════════════
            
            if 'region' in df_filtered.columns:
                regions_in_df = df_filtered['region'].unique().tolist()
                
                for region in regions_in_df:
                    if region.lower() in question_lower:
                        df_filtered = df_filtered[df_filtered['region'] == region]
                        results.append(f"🗺️ Filtré sur la région : {region}")
                        break
            
            # ═══════════════════════════════════════════════════════════
            # ANALYSE DES TRAINS ANNULÉS
            # ═══════════════════════════════════════════════════════════
            
            if 'annulé' in question_lower or 'annule' in question_lower or 'supprimé' in question_lower:
                if 'nb_trains_annules' in df_filtered.columns:
                    total_annules = df_filtered['nb_trains_annules'].sum()
                    total_programmes = df_filtered['nb_trains_programmes'].sum() if 'nb_trains_programmes' in df_filtered.columns else None
                    
                    results.append(f"\n❌ **Trains annulés** : {int(total_annules):,}")
                    
                    if total_programmes:
                        taux_annulation = (total_annules / total_programmes * 100) if total_programmes > 0 else 0
                        results.append(f"📊 Taux d'annulation : {taux_annulation:.2f}%")
                        results.append(f"🚆 Trains programmés : {int(total_programmes):,}")
                
                elif 'nb_train_annule' in df_filtered.columns:
                    total_annules = df_filtered['nb_train_annule'].sum()
                    results.append(f"\n❌ **Trains annulés** : {int(total_annules):,}")
            
            # ═══════════════════════════════════════════════════════════
            # RÉGULARITÉ
            # ═══════════════════════════════════════════════════════════
            
            if 'régularité' in question_lower or 'regularite' in question_lower or 'ponctualité' in question_lower:
                if 'taux_regularite' in df_filtered.columns:
                    avg_reg = df_filtered['taux_regularite'].mean()
                    min_reg = df_filtered['taux_regularite'].min()
                    max_reg = df_filtered['taux_regularite'].max()
                    
                    results.append(f"\n📈 **Régularité moyenne** : {avg_reg:.2f}%")
                    results.append(f"📉 Régularité min : {min_reg:.2f}%")
                    results.append(f"📊 Régularité max : {max_reg:.2f}%")
            
            # ═══════════════════════════════════════════════════════════
            # STATISTIQUES GÉNÉRALES
            # ═══════════════════════════════════════════════════════════
            
            if len(results) == 0 or 'résumé' in question_lower or 'resume' in question_lower:
                results.append(f"\n📊 **Nombre d'enregistrements** : {len(df_filtered):,}")
                
                if 'taux_regularite' in df_filtered.columns:
                    avg_reg = df_filtered['taux_regularite'].mean()
                    results.append(f"📈 **Régularité moyenne** : {avg_reg:.2f}%")
                
                if 'nb_trains_annules' in df_filtered.columns:
                    total_annules = df_filtered['nb_trains_annules'].sum()
                    results.append(f"❌ **Trains annulés** : {int(total_annules):,}")
            
            # ═══════════════════════════════════════════════════════════
            # TOP/PIRES RÉGIONS
            # ═══════════════════════════════════════════════════════════
            
            if ('meilleur' in question_lower or 'top' in question_lower) and 'region' in df_filtered.columns:
                if 'taux_regularite' in df_filtered.columns:
                    top_regions = df_filtered.groupby('region')['taux_regularite'].mean().nlargest(5)
                    results.append("\n🏆 **Top 5 des meilleures régions** :")
                    for i, (region, score) in enumerate(top_regions.items(), 1):
                        results.append(f"  {i}. {region} : {score:.2f}%")
            
            if ('pire' in question_lower or 'worst' in question_lower) and 'region' in df_filtered.columns:
                if 'taux_regularite' in df_filtered.columns:
                    worst_regions = df_filtered.groupby('region')['taux_regularite'].mean().nsmallest(5)
                    results.append("\n📉 **5 pires régions** :")
                    for i, (region, score) in enumerate(worst_regions.items(), 1):
                        results.append(f"  {i}. {region} : {score:.2f}%")
            
            # ═══════════════════════════════════════════════════════════
            # RETOUR
            # ═══════════════════════════════════════════════════════════
            
            if len(results) > 0:
                return "\n".join(results)
            else:
                return "Aucune donnée correspondante trouvée pour cette requête."
        
        except Exception as e:
            return f"❌ Erreur lors de l'analyse des données : {str(e)}"
    
    def _create_system_prompt(self, data_analysis: str = "") -> str:
        """Crée le prompt système pour l'agent"""
        
        return f"""Tu es un assistant IA expert en analyse de données ferroviaires TER en France.

**CONTEXTE DES DONNÉES :**
{self.data_context}

**RÉSULTATS DE L'ANALYSE DES DONNÉES :**
{data_analysis}

**INSTRUCTIONS :**
1. Utilise PRIORITAIREMENT les résultats de l'analyse des données fournis ci-dessus
2. Réponds en français de manière claire et concise (max 150 mots)
3. Utilise des émojis (📊 🗺️ 📈 ✅ ❌ 🌦️)
4. Structure tes réponses avec des bullet points si pertinent
5. Si les données montrent un résultat, commence TOUJOURS par le chiffre principal
6. Ajoute du contexte et de l'interprétation aux chiffres

**EXEMPLE DE BONNE RÉPONSE :**

Question : "Combien de trains ont été annulés en avril 2020 en Bretagne ?"

Réponse :
"❌ **En avril 2020 en Bretagne, 3 245 trains ont été annulés.**

📊 Cela représente un taux d'annulation de **15.8%** (sur 20 523 trains programmés).

⚠️ **Contexte** : Avril 2020 correspond au premier confinement COVID-19, ce qui explique ce taux d'annulation élevé. Les services ont été fortement réduits durant cette période exceptionnelle."

Sois factuel, précis et utile !"""
    
    def ask(self, question: str) -> str:
        """
        Pose une question à l'agent
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Réponse de l'agent
        """
        print(f"\n🤔 Question reçue : {question}")
        
        if not self.api_key:
            return "❌ **Erreur** : Clé API Groq non configurée.\n\nPour configurer :\n1. Créez un compte gratuit sur https://console.groq.com/\n2. Créez une clé API\n3. Ajoutez `GROQ_API_KEY=votre_clé` dans le fichier `.env`"
        
        # 1. ANALYSER LES DONNÉES DIRECTEMENT
        data_analysis = self._analyze_data_for_question(question)
        print(f"📊 Analyse des données : {len(data_analysis)} caractères")
        
        # 2. AJOUTER À L'HISTORIQUE
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        # 3. CRÉER LES MESSAGES POUR L'API
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt(data_analysis)
            }
        ] + self.conversation_history[-6:]  # Garder seulement les 3 derniers échanges
        
        try:
            # 4. APPELER L'API GROQ
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,  # Plus bas pour plus de précision
                "max_tokens": 800,
                "top_p": 1,
                "stream": False
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # 5. TRAITER LA RÉPONSE
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # Ajouter la réponse à l'historique
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                print(f"✅ Réponse générée")
                return assistant_response
            
            else:
                error_msg = f"❌ Erreur API (code {response.status_code})"
                if response.status_code == 401:
                    error_msg += "\n\n🔑 **Clé API invalide**. Vérifiez votre clé sur https://console.groq.com/"
                elif response.status_code == 429:
                    error_msg += "\n\n⏳ **Quota dépassé**. Attendez quelques minutes."
                
                # Retourner au moins l'analyse des données
                return f"{error_msg}\n\n**Données analysées :**\n{data_analysis}"
        
        except Exception as e:
            error_msg = f"❌ **Erreur** : {str(e)}"
            # Retourner au moins l'analyse des données
            return f"{error_msg}\n\n**Données analysées :**\n{data_analysis}"
    
    def reset_conversation(self):
        """Réinitialise l'historique de conversation"""
        self.conversation_history = []
        print("🔄 Historique de conversation réinitialisé")
    
    def get_conversation_length(self) -> int:
        """Retourne le nombre de messages dans l'historique"""
        return len(self.conversation_history)


# ═══════════════════════════════════════════════════════════════════════
# TEST DU MODULE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test du module ai_agent.py\n")
    
    # Créer un DataFrame de test réaliste
    test_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=365 * 3, freq='D'),
        'region': ['Bretagne'] * 365 + ['Normandie'] * 365 + ['Bretagne'] * 365,
        'taux_regularite': [85 + i*0.01 for i in range(365 * 3)],
        'nb_trains_programmes': [250] * (365 * 3),
        'nb_trains_annules': [20] * (365 * 3)
    })
    
    # Simuler avril 2020 avec plus d'annulations (COVID)
    mask_avril_2020 = (test_df['date'].dt.year == 2020) & (test_df['date'].dt.month == 4)
    test_df.loc[mask_avril_2020, 'nb_trains_annules'] = 150
    
    print(f"📊 DataFrame de test : {len(test_df)} lignes\n")
    
    try:
        agent = TERAnalysisAgent(test_df)
        
        print("="*70)
        print("TEST : avril 2020, en bretagne, combien de trains ont été annulés")
        print("="*70)
        response = agent.ask("avril 2020, en bretagne, combien de trains ont été annulés")
        print(f"\n📝 Réponse :\n{response}\n")
        
        print("="*70)
        print("✅ TEST RÉUSSI")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
