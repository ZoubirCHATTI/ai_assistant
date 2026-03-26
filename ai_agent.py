# -*- coding: utf-8 -*-
"""
Agent IA pour l'analyse conversationnelle des données TER
Utilise LangGraph pour une architecture moderne et stable
"""

import streamlit as st
import pandas as pd
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import Config


# ═══════════════════════════════════════════════════════════════════════
# DÉFINITION DE L'ÉTAT DU GRAPH
# ═══════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """État partagé dans le graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    df: pd.DataFrame  # Référence au DataFrame


# ═══════════════════════════════════════════════════════════════════════
# CLASSE DE L'AGENT TER
# ═══════════════════════════════════════════════════════════════════════
class TERAnalysisAgent:
    """Agent IA pour analyser les données TER avec LangGraph"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = ChatMistralAI(
            model=Config.MISTRAL_MODEL,
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0
        )
        
        # Créer les outils
        self.tools = self._create_tools()
        
        # Lier les outils au LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Créer le graph
        self.graph = self._create_graph()
    
    def _create_tools(self):  # ← ATTENTION : 4 espaces d'indentation
        """Crée les outils d'analyse disponibles pour l'agent"""
        
        df = self.df  # Référence locale pour les closures
        
        @tool
        def calculer_regularite_globale() -> str:
            """Calcule le taux de régularité global sur toutes les données."""
            if 'taux_regularite' in df.columns:
                avg = df['taux_regularite'].mean()
                median = df['taux_regularite'].median()
                min_val = df['taux_regularite'].min()
                max_val = df['taux_regularite'].max()
                return (f"Statistiques de régularité globale :\n"
                       f"- Moyenne : {avg:.2f}%\n"
                       f"- Médiane : {median:.2f}%\n"
                       f"- Minimum : {min_val:.2f}%\n"
                       f"- Maximum : {max_val:.2f}%")
            return "Colonne 'taux_regularite' non trouvée dans les données"
        
        @tool
        def regularite_par_region(region: str) -> str:
            """
            Calcule le taux de régularité pour une région spécifique.
            
            Args:
                region: Nom de la région à analyser (ex: 'Île-de-France')
            """
            if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                return "Colonnes nécessaires non trouvées"
            
            df_region = df[df['region'].str.contains(region, case=False, na=False)]
            if df_region.empty:
                regions_disponibles = df['region'].unique()[:5]
                return (f"Aucune donnée trouvée pour la région '{region}'.\n"
                       f"Régions disponibles : {', '.join(regions_disponibles)}")
            
            avg = df_region['taux_regularite'].mean()
            nb_records = len(df_region)
            min_val = df_region['taux_regularite'].min()
            max_val = df_region['taux_regularite'].max()
            
            return (f"Région '{region}' :\n"
                   f"- Régularité moyenne : {avg:.2f}%\n"
                   f"- Nombre d'enregistrements : {nb_records}\n"
                   f"- Régularité min : {min_val:.2f}%\n"
                   f"- Régularité max : {max_val:.2f}%")
        
        @tool
        def top_regions_regulieres(n: int = 5) -> str:
            """
            Liste les N régions les plus régulières.
            
            Args:
                n: Nombre de régions à lister (par défaut 5)
            """
            try:
                # Vérifier les colonnes nécessaires
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée dans les données"
                
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée dans les données"
                
                # Vérifier qu'il y a des données
                if len(df) == 0:
                    return "❌ Le dataset est vide"
                
                # Supprimer les valeurs nulles
                df_clean = df[['region', 'taux_regularite']].dropna()
                
                if len(df_clean) == 0:
                    return "❌ Aucune donnée valide après nettoyage"
                
                # Calculer la moyenne par région
                regularite_par_region = df_clean.groupby('region')['taux_regularite'].mean()
                
                if len(regularite_par_region) == 0:
                    return "❌ Impossible de calculer la régularité par région"
                
                # Récupérer les N meilleures
                n_regions = min(n, len(regularite_par_region))
                top = regularite_par_region.nlargest(n_regions)
                
                result = f"🏆 **Top {n_regions} régions les plus régulières :**\n\n"
                
                for i, (region, taux) in enumerate(top.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    result += f"{emoji} **{region}** : {taux:.2f}%\n"
                
                # Ajouter un contexte
                result += f"\n📊 **Contexte :** {len(regularite_par_region)} régions analysées au total"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur lors de l'analyse des meilleures régions : {str(e)}"
                
        @tool
        def pires_regions(n: int = 5) -> str:
            """
            Liste les N régions avec la pire régularité.
            
            Args:
                n: Nombre de régions à lister (par défaut 5)
            """
            try:
                # Vérifier les colonnes nécessaires
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée dans les données"
                
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée dans les données"
                
                # Vérifier qu'il y a des données
                if len(df) == 0:
                    return "❌ Le dataset est vide"
                
                # Supprimer les valeurs nulles
                df_clean = df[['region', 'taux_regularite']].dropna()
                
                if len(df_clean) == 0:
                    return "❌ Aucune donnée valide après nettoyage"
                
                # Calculer la moyenne par région
                regularite_par_region = df_clean.groupby('region')['taux_regularite'].mean()
                
                if len(regularite_par_region) == 0:
                    return "❌ Impossible de calculer la régularité par région"
                
                # Récupérer les N pires
                n_regions = min(n, len(regularite_par_region))  # S'assurer qu'on ne demande pas plus que disponible
                bottom = regularite_par_region.nsmallest(n_regions)
                
                result = f"⚠️ **Top {n_regions} régions avec la pire régularité :**\n\n"
                
                for i, (region, taux) in enumerate(bottom.items(), 1):
                    result += f"{i}. **{region}** : {taux:.2f}% ⚠️\n"
                
                # Ajouter un contexte
                result += f"\n📊 **Contexte :** {len(regularite_par_region)} régions analysées au total"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur lors de l'analyse des pires régions : {str(e)}"
                
        @tool
        def statistiques_trains() -> str:
            """Donne des statistiques complètes sur le nombre de trains."""
            stats = []
            
            if 'nombre_trains_prevus' in df.columns:
                total_prevus = df['nombre_trains_prevus'].sum()
                stats.append(f"🚂 Total trains prévus : {total_prevus:,.0f}")
            
            if 'nombre_trains_circules' in df.columns:
                total_circules = df['nombre_trains_circules'].sum()
                stats.append(f"✅ Total trains circulés : {total_circules:,.0f}")
                
                if 'nombre_trains_prevus' in df.columns:
                    taux_circulation = (total_circules / total_prevus * 100) if total_prevus > 0 else 0
                    stats.append(f"📊 Taux de circulation : {taux_circulation:.2f}%")
            
            if 'nombre_trains_supprimes' in df.columns:
                total_supprimes = df['nombre_trains_supprimes'].sum()
                stats.append(f"❌ Total trains supprimés : {total_supprimes:,.0f}")
            
            if 'nombre_trains_retard' in df.columns:
                total_retards = df['nombre_trains_retard'].sum()
                stats.append(f"⏰ Total trains en retard : {total_retards:,.0f}")
            
            return "\n".join(stats) if stats else "Données de trains non disponibles"
        
        @tool
        def evolution_temporelle() -> str:
            """Analyse l'évolution de la régularité dans le temps."""
            if 'annee' in df.columns and 'mois' in df.columns and 'taux_regularite' in df.columns:
                evolution = df.groupby(['annee', 'mois'])['taux_regularite'].mean()
                if len(evolution) >= 2:
                    first = evolution.iloc[0]
                    last = evolution.iloc[-1]
                    diff = last - first
                    trend = "amélioration" if diff > 0 else "dégradation"
                    return (f"📈 Évolution de la régularité :\n"
                           f"- Période initiale : {first:.2f}%\n"
                           f"- Période récente : {last:.2f}%\n"
                           f"- Tendance : {trend} de {abs(diff):.2f} points")
            return "Données temporelles insuffisantes"
        
        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions présentes dans les données."""
            if 'region' not in df.columns:
                return "Colonne 'region' non trouvée"
            
            regions = sorted(df['region'].unique())
            result = f"🗺️ {len(regions)} régions disponibles :\n\n"
            for region in regions:
                nb = len(df[df['region'] == region])
                result += f"- {region} ({nb} enregistrements)\n"
            return result
        
        # ═══════════════════════════════════════════════════════════════
        # OUTILS MÉTÉO
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def verifier_donnees_meteo() -> str:
            """Vérifie si les données météo sont disponibles."""
            weather_cols = ['temperature_mean', 'precipitation', 'snow', 'wind_speed', 'weather_severity_score']
            available = [col for col in weather_cols if col in df.columns]
            
            if not available:
                return ("❌ Aucune donnée météo disponible.\n"
                       "Enrichissez le dataset dans la page '🌦️ Analyse Météo'")
            
            result = f"✅ {len(available)}/{len(weather_cols)} colonnes météo disponibles\n"
            result += f"Colonnes : {', '.join(available)}"
            return result
        
        @tool
        def analyser_impact_meteo_global() -> str:
            """Analyse l'impact global de la météo sur la régularité."""
            if 'weather_severity_score' not in df.columns or 'taux_regularite' not in df.columns:
                return "❌ Données météo ou régularité manquantes"
            
            df_clean = df.dropna(subset=['weather_severity_score', 'taux_regularite'])
            
            if len(df_clean) < 10:
                return "❌ Pas assez de données"
            
            df_clean['meteo_cat'] = pd.cut(
                df_clean['weather_severity_score'],
                bins=[-1, 20, 40, 60, 100],
                labels=['Bonne', 'Correcte', 'Difficile', 'Extrême']
            )
            
            result = "🌦️ Impact météo sur la régularité :\n\n"
            for cat in ['Bonne', 'Correcte', 'Difficile', 'Extrême']:
                data = df_clean[df_clean['meteo_cat'] == cat]
                if len(data) > 0:
                    avg = data['taux_regularite'].mean()
                    result += f"- Météo {cat}: {avg:.2f}% ({len(data)} jours)\n"
            
            return result
        
        @tool
        def impact_neige_sur_regularite() -> str:
            """Analyse l'impact de la neige sur la régularité."""
            if 'snow' not in df.columns or 'taux_regularite' not in df.columns:
                return "❌ Données de neige non disponibles"
            
            df_clean = df.dropna(subset=['snow', 'taux_regularite'])
            avec_neige = df_clean[df_clean['snow'] > 0]
            sans_neige = df_clean[df_clean['snow'] == 0]
            
            if len(avec_neige) == 0:
                return "✅ Aucun épisode neigeux détecté"
            
            reg_avec = avec_neige['taux_regularite'].mean()
            reg_sans = sans_neige['taux_regularite'].mean()
            diff = reg_sans - reg_avec
            
            result = f"❄️ Impact de la neige :\n"
            result += f"- Sans neige: {reg_sans:.2f}%\n"
            result += f"- Avec neige: {reg_avec:.2f}%\n"
            result += f"- Perte: {diff:.2f} points\n"
            result += f"- Jours avec neige: {len(avec_neige)}"
            
            return result
        
        @tool
        def impact_vent_fort() -> str:
            """Analyse l'impact des vents forts (>90 km/h)."""
            if 'wind_gusts' not in df.columns or 'taux_regularite' not in df.columns:
                return "❌ Données de vent non disponibles"
            
            df_clean = df.dropna(subset=['wind_gusts', 'taux_regularite'])
            vent_fort = df_clean[df_clean['wind_gusts'] > 90]
            vent_normal = df_clean[df_clean['wind_gusts'] <= 90]
            
            if len(vent_fort) == 0:
                return "✅ Aucun vent fort (>90 km/h) détecté"
            
            reg_fort = vent_fort['taux_regularite'].mean()
            reg_normal = vent_normal['taux_regularite'].mean()
            diff = reg_normal - reg_fort
            
            result = f"💨 Impact du vent fort :\n"
            result += f"- Vent normal: {reg_normal:.2f}%\n"
            result += f"- Vent fort: {reg_fort:.2f}%\n"
            result += f"- Perte: {diff:.2f} points\n"
            result += f"- Jours avec vent fort: {len(vent_fort)}"
            
            return result
        
        @tool
        def jours_meteo_extreme() -> str:
            """Liste les jours avec météo extrême (score > 70)."""
            if 'weather_severity_score' not in df.columns:
                return "❌ Score de sévérité non disponible"
            
            extreme = df[df['weather_severity_score'] > 70].copy()
            
            if len(extreme) == 0:
                return "✅ Aucun jour avec météo extrême"
            
            extreme = extreme.sort_values('weather_severity_score', ascending=False).head(10)
            
            result = f"⛈️ {len(extreme)} jours avec météo EXTRÊME (top 10) :\n\n"
            for idx, row in extreme.iterrows():
                date = row.get('date', 'N/A')
                score = row['weather_severity_score']
                reg = row.get('taux_regularite', None)
                
                date_str = date.strftime('%d/%m/%Y') if hasattr(date, 'strftime') else str(date)
                result += f"📅 {date_str} - Score: {score:.0f}"
                if reg:
                    result += f" - Régularité: {reg:.1f}%"
                result += "\n"
            
            return result
        
        # Retourner TOUS les outils
        return [
            calculer_regularite_globale,
            regularite_par_region,
            top_regions_regulieres,
            pires_regions,
            statistiques_trains,
            evolution_temporelle,
            liste_regions_disponibles,
            verifier_donnees_meteo,
            analyser_impact_meteo_global,
            impact_neige_sur_regularite,
            impact_vent_fort,
            jours_meteo_extreme
        ]
    
    def _create_graph(self):
        """Crée le graph LangGraph"""
        
        workflow = StateGraph(AgentState)
        
        def call_model(state: AgentState):
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        tool_node = ToolNode(self.tools)
        
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
 
    
    def ask(self, question: str) -> str:
        """
        Pose une question à l'agent
        
        Args:
            question: Question en langage naturel
            
        Returns:
            Réponse de l'agent
        """
        try:
            # Créer le message initial
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "df": self.df
            }
            
            # Exécuter le graph
            result = self.graph.invoke(initial_state)
            
            # Extraire la réponse finale
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(final_message)
            
        except Exception as e:
            return f"❌ Erreur lors du traitement de la question : {str(e)}"
@tool
def analyser_impact_meteo() -> str:
    """Analyse l'impact global des conditions météorologiques sur les retards et la régularité."""
    
    # Cette fonction sera appelée depuis l'app principale
    return ("Pour une analyse météo complète, rendez-vous dans la section "
            "'🌦️ Analyse Météo' du menu principal.")

@tool
def causes_retards_meteo() -> str:
    """Détermine si les retards sont liés à des conditions météorologiques."""
    
    if 'weather_severity_score' not in df.columns:
        return ("❌ Données météo non disponibles. Veuillez d'abord enrichir "
                "le dataset avec les données météo dans la section dédiée.")
    
    # Filtrer les jours avec forte sévérité météo
    df_meteo_severe = df[df['weather_severity_score'] > 60]
    
    if df_meteo_severe.empty:
        return "✅ Aucune condition météo extrême détectée dans la période analysée."
    
    # Analyser les retards lors de météo sévère
    if 'nombre_trains_retard' in df_meteo_severe.columns:
        total_jours_severe = len(df_meteo_severe)
        retards_meteo_severe = df_meteo_severe['nombre_trains_retard'].sum()
        retards_moyenne_severe = df_meteo_severe['nombre_trains_retard'].mean()
        retards_moyenne_normale = df[df['weather_severity_score'] <= 60]['nombre_trains_retard'].mean()
        
        augmentation = ((retards_moyenne_severe - retards_moyenne_normale) / retards_moyenne_normale * 100)
        
        return (f"🌨️ Impact de la météo sur les retards :\n\n"
                f"- {total_jours_severe} jours avec conditions météo sévères\n"
                f"- {retards_meteo_severe:,.0f} trains en retard lors de ces jours\n"
                f"- Moyenne de {retards_moyenne_severe:.1f} retards/jour (météo sévère)\n"
                f"- Moyenne de {retards_moyenne_normale:.1f} retards/jour (météo normale)\n"
                f"- Augmentation de {augmentation:+.1f}% des retards en conditions météo difficiles")
    
    return "Données insuffisantes pour l'analyse"
