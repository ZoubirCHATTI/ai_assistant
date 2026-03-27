# ai_agent.py

}"
# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA pour l'analyse des données TER avec génération de graphiques
"""

from mistralai import Mistral
import pandas as pd
import json
from config import Config


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER et générer des graphiques"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'agent avec les données TER
        
        Args:
            df: DataFrame contenant les données TER
        """
        if not Config.MISTRAL_API_KEY:
            raise ValueError("❌ MISTRAL_API_KEY n'est pas configurée")
        
        self.client = Mistral(api_key=Config.MISTRAL_API_KEY)
        self.model = Config.MISTRAL_MODEL
        self.df = df
        
        # Préparer un résumé des données pour le contexte
        self.data_context = self._prepare_data_context()
        
        print("✅ Agent IA initialisé avec succès")
        print(f"   Modèle : {self.model}")
        print(f"   Données : {len(df)} lignes, {len(df.columns)} colonnes")
    
    def _prepare_data_context(self) -> str:
        """Prépare un résumé des données pour le contexte de l'IA"""
        
        context_parts = [
            f"Dataset TER : {len(self.df)} enregistrements",
            f"Colonnes disponibles : {', '.join(self.df.columns.tolist())}",
        ]
        
        # Ajouter des statistiques de base
        if 'taux_regularite' in self.df.columns:
            avg_reg = self.df['taux_regularite'].mean()
            context_parts.append(f"Taux de régularité moyen : {avg_reg:.2f}%")
        
        if 'region' in self.df.columns:
            regions = self.df['region'].unique()
            context_parts.append(f"Régions : {', '.join(regions[:10])}" + 
                               (" ..." if len(regions) > 10 else ""))
        
        if 'date' in self.df.columns:
            date_min = self.df['date'].min()
            date_max = self.df['date'].max()
            context_parts.append(f"Période : du {date_min} au {date_max}")
        
        return "\n".join(context_parts)
    
    def _analyze_data(self, question: str) -> dict:
        """
        Analyse les données selon la question
        
        Returns:
            dict avec 'data' (résultats) et 'summary' (texte)
        """
        
        # Créer différents types d'analyses selon la question
        question_lower = question.lower()
        
        result = {
            'data': None,
            'summary': '',
            'chart_type': None,
            'chart_data': None
        }
        
        try:
            # Régularité moyenne globale
            if any(word in question_lower for word in ['moyenne', 'globale', 'général']):
                if 'taux_regularite' in self.df.columns:
                    avg = self.df['taux_regularite'].mean()
                    result['summary'] = f"La régularité moyenne est de {avg:.2f}%"
                    result['data'] = {'moyenne': avg}
            
            # Régularité par région
            elif any(word in question_lower for word in ['région', 'regions', 'compare']):
                if 'region' in self.df.columns and 'taux_regularite' in self.df.columns:
                    by_region = self.df.groupby('region')['taux_regularite'].mean().sort_values(ascending=False)
                    
                    result['chart_type'] = 'bar'
                    result['chart_data'] = {
                        'x': by_region.index.tolist(),
                        'y': by_region.values.tolist(),
                        'title': 'Régularité moyenne par région',
                        'xlabel': 'Région',
                        'ylabel': 'Taux de régularité (%)'
                    }
                    
                    top3 = by_region.head(3)
                    result['summary'] = "Top 3 des régions :\n" + "\n".join(
                        [f"{i+1}. {region}: {taux:.2f}%" 
                         for i, (region, taux) in enumerate(top3.items())]
                    )
                    result['data'] = by_region.to_dict()
            
            # Top / meilleures régions
            elif any(word in question_lower for word in ['meilleur', 'top', 'première']):
                if 'region' in self.df.columns and 'taux_regularite' in self.df.columns:
                    n = 5  # Par défaut
                    if 'top 10' in question_lower or '10' in question_lower:
                        n = 10
                    elif 'top 3' in question_lower or '3' in question_lower:
                        n = 3
                    
                    top_regions = self.df.groupby('region')['taux_regularite'].mean().nlargest(n)
                    
                    result['chart_type'] = 'bar'
                    result['chart_data'] = {
                        'x': top_regions.index.tolist(),
                        'y': top_regions.values.tolist(),
                        'title': f'Top {n} des régions les plus ponctuelles',
                        'xlabel': 'Région',
                        'ylabel': 'Taux de régularité (%)',
                        'color': 'green'
                    }
                    
                    result['summary'] = f"Top {n} des régions les plus ponctuelles :\n" + "\n".join(
                        [f"{i+1}. {region}: {taux:.2f}%" 
                         for i, (region, taux) in enumerate(top_regions.items())]
                    )
                    result['data'] = top_regions.to_dict()
            
            # Pires régions
            elif any(word in question_lower for word in ['pire', 'mauvais', 'dernière', 'worst']):
                if 'region' in self.df.columns and 'taux_regularite' in self.df.columns:
                    n = 5
                    if '10' in question_lower:
                        n = 10
                    elif '3' in question_lower:
                        n = 3
                    
                    worst_regions = self.df.groupby('region')['taux_regularite'].mean().nsmallest(n)
                    
                    result['chart_type'] = 'bar'
                    result['chart_data'] = {
                        'x': worst_regions.index.tolist(),
                        'y': worst_regions.values.tolist(),
                        'title': f'Top {n} des régions les moins ponctuelles',
                        'xlabel': 'Région',
                        'ylabel': 'Taux de régularité (%)',
                        'color': 'red'
                    }
                    
                    result['summary'] = f"Top {n} des régions les moins ponctuelles :\n" + "\n".join(
                        [f"{i+1}. {region}: {taux:.2f}%" 
                         for i, (region, taux) in enumerate(worst_regions.items())]
                    )
                    result['data'] = worst_regions.to_dict()
            
            # Évolution temporelle
            elif any(word in question_lower for word in ['évolution', 'evolution', 'temps', 'tendance']):
                if 'date' in self.df.columns and 'taux_regularite' in self.df.columns:
                    by_date = self.df.groupby('date')['taux_regularite'].mean().sort_index()
                    
                    result['chart_type'] = 'line'
                    result['chart_data'] = {
                        'x': [str(d) for d in by_date.index.tolist()],
                        'y': by_date.values.tolist(),
                        'title': 'Évolution de la régularité dans le temps',
                        'xlabel': 'Date',
                        'ylabel': 'Taux de régularité (%)'
                    }
                    
                    trend = "hausse" if by_date.iloc[-1] > by_date.iloc[0] else "baisse"
                    result['summary'] = f"Évolution : tendance à la {trend}\n"
                    result['summary'] += f"Début : {by_date.iloc[0]:.2f}%\n"
                    result['summary'] += f"Fin : {by_date.iloc[-1]:.2f}%"
                    result['data'] = by_date.to_dict()
            
            # Impact météo (si colonnes disponibles)
            elif any(word in question_lower for word in ['météo', 'meteo', 'neige', 'pluie', 'vent']):
                weather_cols = [col for col in self.df.columns if any(
                    w in col.lower() for w in ['neige', 'pluie', 'vent', 'temp']
                )]
                
                if weather_cols and 'taux_regularite' in self.df.columns:
                    # Analyser l'impact de la première colonne météo trouvée
                    weather_col = weather_cols[0]
                    
                    # Créer des catégories
                    if 'neige' in weather_col.lower() or 'pluie' in weather_col.lower():
                        self.df['meteo_category'] = pd.cut(
                            self.df[weather_col], 
                            bins=[0, 0.1, 5, 100], 
                            labels=['Pas de précipitations', 'Légères', 'Fortes']
                        )
                    else:
                        self.df['meteo_category'] = pd.cut(
                            self.df[weather_col], 
                            bins=3, 
                            labels=['Faible', 'Moyen', 'Fort']
                        )
                    
                    by_weather = self.df.groupby('meteo_category')['taux_regularite'].mean()
                    
                    result['chart_type'] = 'bar'
                    result['chart_data'] = {
                        'x': [str(x) for x in by_weather.index.tolist()],
                        'y': by_weather.values.tolist(),
                        'title': f'Impact de {weather_col} sur la régularité',
                        'xlabel': weather_col,
                        'ylabel': 'Taux de régularité (%)'
                    }
                    
                    result['summary'] = f"Impact de {weather_col} :\n" + "\n".join(
                        [f"- {cat}: {taux:.2f}%" 
                         for cat, taux in by_weather.items()]
                    )
                    result['data'] = by_weather.to_dict()
        
        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse : {e}")
            result['summary'] = "Je n'ai pas pu analyser cette dimension des données."
        
        return result
    
    def ask(self, question: str) -> dict:
        """
        Pose une question à l'agent
        
        Args:
            question: Question en langage naturel
            
        Returns:
            dict avec 'text' (réponse), 'chart_type', 'chart_data'
        """
        
        print(f"\n🤔 Question : {question}")
        
        # Analyser les données
        analysis = self._analyze_data(question)
        
        # Construire le prompt pour l'IA
        system_prompt = f"""Tu es un assistant spécialisé dans l'analyse des données de régularité des trains TER en France.

CONTEXTE DES DONNÉES :
{self.data_context}

ANALYSE EFFECTUÉE :
{analysis['summary'] if analysis['summary'] else 'Aucune analyse spécifique'}

Réponds à la question de l'utilisateur de manière claire et concise en français.
Si des données chiffrées sont disponibles, cite-les.
Sois précis et factuel."""
        
        try:
            # Appeler l'API Mistral
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer_text = response.choices[0].message.content
            
            print(f"✅ Réponse générée")
            
            return {
                'text': answer_text,
                'chart_type': analysis.get('chart_type'),
                'chart_data': analysis.get('chart_data')
            }
        
        except Exception as e:
            print(f"❌ Erreur API Mistral : {e}")
            
            # Réponse de secours avec les données analysées
            if analysis['summary']:
                return {
                    'text': analysis['summary'],
                    'chart_type': analysis.get('chart_type'),
                    'chart_data': analysis.get('chart_data')
                }
            else:
                raise e
