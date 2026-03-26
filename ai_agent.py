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
    
    def _create_tools(self):
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
                region: Nom de la région à analyser (ex: 'Île-de-France', 'Auvergne-Rhône-Alpes')
            """
            if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                return "Colonnes nécessaires non trouvées"
            
            df_region = df[df['region'].str.contains(region, case=False, na=False)]
            if df_region.empty:
                regions_disponibles = df['region'].unique()[:5]
                return (f"Aucune donnée trouvée pour la région '{region}'.\n"
                       f"Régions disponibles (exemples) : {', '.join(regions_disponibles)}")
            
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
            if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                return "Colonnes nécessaires non trouvées"
            
            top = df.groupby('region')['taux_regularite'].mean().nlargest(n)
            result = f"🏆 Top {n} régions les plus régulières :\n\n"
            for i, (region, taux) in enumerate(top.items(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                result += f"{emoji} {region} : {taux:.2f}%\n"
            return result
        
        @tool
        def pires_regions(n: int = 5) -> str:
            """
            Liste les N régions avec la pire régularité.
            
            Args:
                n: Nombre de régions à lister (par défaut 5)
            """
            if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                return "Colonnes nécessaires non trouvées"
            
            bottom = df.groupby('region')['taux_regularite'].mean().nsmallest(n)
            result = f"⚠️ Top {n} régions avec la pire régularité :\n\n"
            for i, (region, taux) in enumerate(bottom.items(), 1):
                result += f"{i}. {region} : {taux:.2f}% ⚠️\n"
            return result
        
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
                
                if 'nombre_trains_prevus' in df.columns:
                    taux_suppression = (total_supprimes / total_prevus * 100) if total_prevus > 0 else 0
                    stats.append(f"📉 Taux de suppression : {taux_suppression:.2f}%")
            
            if 'nombre_trains_retard' in df.columns:
                total_retards = df['nombre_trains_retard'].sum()
                stats.append(f"⏰ Total trains en retard : {total_retards:,.0f}")
                
                if 'nombre_trains_circules' in df.columns:
                    taux_retard = (total_retards / total_circules * 100) if total_circules > 0 else 0
                    stats.append(f"📊 Taux de retard : {taux_retard:.2f}%")
            
            if 'nombre_trains_a_l_heure' in df.columns:
                total_a_l_heure = df['nombre_trains_a_l_heure'].sum()
                stats.append(f"✅ Total trains à l'heure : {total_a_l_heure:,.0f}")
            
            return "\n".join(stats) if stats else "Données de trains non disponibles"
        
        @tool
        def evolution_temporelle() -> str:
            """Analyse l'évolution de la régularité dans le temps."""
            if 'date' not in df.columns or 'taux_regularite' not in df.columns:
                if 'annee' in df.columns and 'mois' in df.columns:
                    # Analyse par année/mois
                    evolution = df.groupby(['annee', 'mois'])['taux_regularite'].mean()
                    first = evolution.iloc[0]
                    last = evolution.iloc[-1]
                    diff = last - first
                    trend = "amélioration" if diff > 0 else "dégradation"
                    return (f"📈 Évolution de la régularité :\n"
                           f"- Période initiale : {first:.2f}%\n"
                           f"- Période récente : {last:.2f}%\n"
                           f"- Tendance : {trend} de {abs(diff):.2f} points")
                return "Données temporelles insuffisantes pour l'analyse"
            
            # Tri par date
            df_sorted = df.sort_values('date')
            
            # Première et dernière période
            first_month_avg = df_sorted.head(30)['taux_regularite'].mean()
            last_month_avg = df_sorted.tail(30)['taux_regularite'].mean()
            evolution = last_month_avg - first_month_avg
            
            trend = "améliorée" if evolution > 0 else "dégradée"
            
            # Meilleur et pire mois
            monthly_avg = df.groupby(pd.Grouper(key='date', freq='M'))['taux_regularite'].mean()
            best_month = monthly_avg.idxmax()
            worst_month = monthly_avg.idxmin()
            
            return (f"📈 Évolution temporelle de la régularité :\n\n"
                   f"- Début de période : {first_month_avg:.2f}%\n"
                   f"- Fin de période : {last_month_avg:.2f}%\n"
                   f"- Tendance : {trend} ({evolution:+.2f} points)\n\n"
                   f"🏆 Meilleur mois : {best_month.strftime('%B %Y')} ({monthly_avg.max():.2f}%)\n"
                   f"⚠️ Pire mois : {worst_month.strftime('%B %Y')} ({monthly_avg.min():.2f}%)")
        
        @tool
        def comparer_periodes(annee1: int, annee2: int) -> str:
            """
            Compare la régularité entre deux années.
            
            Args:
                annee1: Première année à comparer
                annee2: Deuxième année à comparer
            """
            if 'annee' not in df.columns or 'taux_regularite' not in df.columns:
                return "Colonne 'annee' ou 'taux_regularite' non trouvée"
            
            df_y1 = df[df['annee'] == annee1]
            df_y2 = df[df['annee'] == annee2]
            
            if df_y1.empty or df_y2.empty:
                return f"Données insuffisantes pour {annee1} ou {annee2}"
            
            avg1 = df_y1['taux_regularite'].mean()
            avg2 = df_y2['taux_regularite'].mean()
            diff = avg2 - avg1
            
            evolution = "amélioration" if diff > 0 else "dégradation"
            
            return (f"📊 Comparaison {annee1} vs {annee2} :\n\n"
                   f"- {annee1} : {avg1:.2f}%\n"
                   f"- {annee2} : {avg2:.2f}%\n"
                   f"- Évolution : {evolution} de {abs(diff):.2f} points ({diff:+.2f}%)")
        
        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions présentes dans les données."""
            if 'region' not in df.columns:
                return "Colonne 'region' non trouvée"
            
            regions = sorted(df['region'].unique())
            nb_regions = len(regions)
            
            result = f"🗺️ {nb_regions} régions disponibles dans les données :\n\n"
            for region in regions:
                nb_records = len(df[df['region'] == region])
                result += f"- {region} ({nb_records} enregistrements)\n"
            
            return result
        
        return [
            calculer_regularite_globale,
            regularite_par_region,
            top_regions_regulieres,
            pires_regions,
            statistiques_trains,
            evolution_temporelle,
            comparer_periodes,
            liste_regions_disponibles
        ]
    
    def _create_graph(self):
        """Crée le graph LangGraph"""
        
        # Définir le graph
        workflow = StateGraph(AgentState)
        
        # Nœud de l'agent
        def call_model(state: AgentState):
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Nœud des outils
        tool_node = ToolNode(self.tools)
        
        # Fonction de routage
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            # Si le LLM appelle un outil, continuer
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Sinon, terminer
            return END
        
        # Ajouter les nœuds au graph
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        # Définir les edges
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
        
        # Compiler le graph
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
