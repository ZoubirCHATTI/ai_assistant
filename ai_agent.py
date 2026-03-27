# ai_agent.py

import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
import re

from config import Config


class AgentState(TypedDict):
    """État de l'agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER avec LangGraph"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df  # Stocker le DataFrame comme attribut de classe
        
        # Convertir la colonne date en datetime si nécessaire
        if 'date' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        self.llm = ChatMistralAI(
            model=Config.MISTRAL_MODEL,
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0
        )
        
        # Créer les outils avec accès au DataFrame
        self.tools = self._create_tools()
        
        # Lier les outils au LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Créer le graph
        self.graph = self._create_graph()
    
    def _create_tools(self):
        """Crée les outils d'analyse disponibles pour l'agent"""
        
        # Référence locale EXPLICITE au DataFrame
        df = self.df
        
        # Vérification de sécurité
        if df is None or len(df) == 0:
            raise ValueError("❌ Le DataFrame est vide ou None !")
        
        print(f"✅ Outils créés avec DataFrame de {len(df)} lignes et {len(df.columns)} colonnes")
        
        # ═══════════════════════════════════════════════════════════════
        # OUTILS DE DEBUG
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def debug_dataframe_info() -> str:
            """Affiche des informations de debug sur le DataFrame."""
            try:
                info = f"📊 **Informations sur le DataFrame :**\n\n"
                info += f"- **Nombre de lignes** : {len(df):,}\n"
                info += f"- **Nombre de colonnes** : {len(df.columns)}\n"
                info += f"- **Colonnes** : {', '.join(df.columns.tolist())}\n\n"
                
                if 'region' in df.columns:
                    nb_regions = df['region'].nunique()
                    regions = df['region'].unique()[:5]
                    info += f"✅ **Colonne 'region'** : {nb_regions} régions uniques\n"
                    info += f"   Exemples : {', '.join(str(r) for r in regions)}\n"
                
                if 'date' in df.columns:
                    date_min = df['date'].min()
                    date_max = df['date'].max()
                    info += f"✅ **Colonne 'date'** : du {date_min} au {date_max}\n"
                
                if 'taux_regularite' in df.columns:
                    avg = df['taux_regularite'].mean()
                    info += f"✅ **Colonne 'taux_regularite'** : Moyenne {avg:.2f}%\n"
                
                return info
                
            except Exception as e:
                return f"❌ Erreur debug : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # NOUVEAUX OUTILS DE FILTRAGE
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def filtrer_par_mois_annee_region(mois: int, annee: int, region: str = None) -> str:
            """
            Filtre les données par mois, année et optionnellement par région.
            Retourne des statistiques complètes.
            
            Args:
                mois: Numéro du mois (1-12)
                annee: Année (ex: 2020)
                region: Nom de la région (optionnel, ex: "Bretagne")
            """
            try:
                if 'date' not in df.columns:
                    return "❌ Colonne 'date' non trouvée"
                
                # Filtrer par date
                df_filtered = df.copy()
                df_filtered = df_filtered[
                    (df_filtered['date'].dt.year == annee) & 
                    (df_filtered['date'].dt.month == mois)
                ]
                
                if len(df_filtered) == 0:
                    return f"❌ Aucune donnée pour {mois}/{annee}"
                
                # Filtrer par région si spécifiée
                if region and 'region' in df.columns:
                    # Recherche insensible à la casse
                    df_filtered = df_filtered[df_filtered['region'].str.lower() == region.lower()]
                    
                    if len(df_filtered) == 0:
                        return f"❌ Aucune donnée pour {region} en {mois}/{annee}"
                
                # Construire le résultat
                mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                
                result = f"📅 **{mois_noms[mois].capitalize()} {annee}"
                if region:
                    result += f" - {region}"
                result += "** :\n\n"
                
                result += f"- **Enregistrements** : {len(df_filtered):,}\n"
                
                # Régularité
                if 'taux_regularite' in df_filtered.columns:
                    avg_reg = df_filtered['taux_regularite'].mean()
                    result += f"- **Régularité moyenne** : {avg_reg:.2f}%\n"
                
                # Trains annulés
                col_annules = None
                if 'nombre_trains_supprimes' in df_filtered.columns:
                    col_annules = 'nombre_trains_supprimes'
                elif 'nb_trains_annules' in df_filtered.columns:
                    col_annules = 'nb_trains_annules'
                
                if col_annules:
                    total_annules = df_filtered[col_annules].sum()
                    result += f"- **Trains annulés** : {int(total_annules):,}\n"
                
                # Trains prévus
                col_prevus = None
                if 'nombre_trains_prevus' in df_filtered.columns:
                    col_prevus = 'nombre_trains_prevus'
                elif 'nb_trains_programmes' in df_filtered.columns:
                    col_prevus = 'nb_trains_programmes'
                
                if col_prevus:
                    total_prevus = df_filtered[col_prevus].sum()
                    result += f"- **Trains prévus** : {int(total_prevus):,}\n"
                    
                    if col_annules:
                        taux_annulation = (total_annules / total_prevus * 100) if total_prevus > 0 else 0
                        result += f"- **Taux d'annulation** : {taux_annulation:.2f}%\n"
                
                # Météo si disponible
                if 'weather_severity_score' in df_filtered.columns:
                    avg_meteo = df_filtered['weather_severity_score'].mean()
                    result += f"\n🌦️ **Score météo moyen** : {avg_meteo:.1f}/100"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def comparer_periodes(mois1: int, annee1: int, mois2: int, annee2: int, region: str = None) -> str:
            """
            Compare deux périodes (mois/année) pour une région donnée.
            
            Args:
                mois1: Mois de la période 1 (1-12)
                annee1: Année de la période 1
                mois2: Mois de la période 2 (1-12)
                annee2: Année de la période 2
                region: Nom de la région (optionnel)
            """
            try:
                if 'date' not in df.columns:
                    return "❌ Colonne 'date' non trouvée"
                
                # Période 1
                df_p1 = df[
                    (df['date'].dt.year == annee1) & 
                    (df['date'].dt.month == mois1)
                ]
                
                # Période 2
                df_p2 = df[
                    (df['date'].dt.year == annee2) & 
                    (df['date'].dt.month == mois2)
                ]
                
                # Filtrer par région
                if region and 'region' in df.columns:
                    df_p1 = df_p1[df_p1['region'].str.lower() == region.lower()]
                    df_p2 = df_p2[df_p2['region'].str.lower() == region.lower()]
                
                if len(df_p1) == 0 or len(df_p2) == 0:
                    return "❌ Données insuffisantes pour la comparaison"
                
                mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                
                result = f"📊 **Comparaison "
                if region:
                    result += f"{region} : "
                result += f"{mois_noms[mois1]} {annee1} vs {mois_noms[mois2]} {annee2}** :\n\n"
                
                # Régularité
                if 'taux_regularite' in df.columns:
                    reg1 = df_p1['taux_regularite'].mean()
                    reg2 = df_p2['taux_regularite'].mean()
                    diff = reg1 - reg2
                    
                    result += f"📈 **Régularité** :\n"
                    result += f"- Période 1 : {reg1:.2f}%\n"
                    result += f"- Période 2 : {reg2:.2f}%\n"
                    result += f"- Différence : {diff:+.2f} points\n"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def analyser_region_complete(region: str) -> str:
            """
            Analyse complète d'une région spécifique.
            
            Args:
                region: Nom de la région (ex: "Bretagne")
            """
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"
                
                # Filtrer par région (insensible à la casse)
                df_region = df[df['region'].str.lower() == region.lower()]
                
                if len(df_region) == 0:
                    return f"❌ Région '{region}' non trouvée"
                
                result = f"🗺️ **Analyse complète : {region}** :\n\n"
                result += f"- **Enregistrements** : {len(df_region):,}\n"
                
                # Régularité
                if 'taux_regularite' in df_region.columns:
                    avg_reg = df_region['taux_regularite'].mean()
                    min_reg = df_region['taux_regularite'].min()
                    max_reg = df_region['taux_regularite'].max()
                    
                    result += f"\n📈 **Régularité** :\n"
                    result += f"- Moyenne : {avg_reg:.2f}%\n"
                    result += f"- Min : {min_reg:.2f}%\n"
                    result += f"- Max : {max_reg:.2f}%\n"
                
                # Trains
                if 'nombre_trains_supprimes' in df_region.columns:
                    total_annules = df_region['nombre_trains_supprimes'].sum()
                    result += f"\n❌ **Trains annulés** : {int(total_annules):,}\n"
                
                # Période couverte
                if 'date' in df_region.columns:
                    date_min = df_region['date'].min()
                    date_max = df_region['date'].max()
                    result += f"\n📅 **Période** : du {date_min.strftime('%d/%m/%Y')} au {date_max.strftime('%d/%m/%Y')}"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def chercher_anomalies_meteo(region: str = None, annee: int = None) -> str:
            """
            Cherche les jours avec des conditions météo extrêmes.
            
            Args:
                region: Nom de la région (optionnel)
                annee: Année à analyser (optionnel)
            """
            try:
                if 'weather_severity_score' not in df.columns:
                    return "❌ Données météo non disponibles"
                
                df_filtered = df.copy()
                
                # Filtrer par région
                if region and 'region' in df.columns:
                    df_filtered = df_filtered[df_filtered['region'].str.lower() == region.lower()]
                
                # Filtrer par année
                if annee and 'date' in df.columns:
                    df_filtered = df_filtered[df_filtered['date'].dt.year == annee]
                
                if len(df_filtered) == 0:
                    return "❌ Aucune donnée correspondante"
                
                # Trouver les jours avec météo extrême (score > 70)
                df_extreme = df_filtered[df_filtered['weather_severity_score'] > 70].copy()
                
                if len(df_extreme) == 0:
                    return "✅ Aucune condition météo extrême détectée"
                
                df_extreme = df_extreme.sort_values('weather_severity_score', ascending=False).head(10)
                
                result = f"⛈️ **{len(df_extreme)} jours avec météo extrême** :\n\n"
                
                for idx, row in df_extreme.iterrows():
                    date_str = row['date'].strftime('%d/%m/%Y') if 'date' in row else 'N/A'
                    score = row['weather_severity_score']
                    reg = row.get('taux_regularite', 'N/A')
                    
                    result += f"- **{date_str}** : Score {score:.0f}/100 | Régularité {reg:.1f}%\n"
                
                return result
                
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # OUTILS EXISTANTS (inchangés)
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def calculer_regularite_globale() -> str:
            """Calcule le taux de régularité global sur toutes les données."""
            try:
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée"
                
                avg = df['taux_regularite'].mean()
                median = df['taux_regularite'].median()
                min_val = df['taux_regularite'].min()
                max_val = df['taux_regularite'].max()
                
                return (f"📊 **Statistiques de régularité globale :**\n\n"
                       f"- **Moyenne** : {avg:.2f}%\n"
                       f"- **Médiane** : {median:.2f}%\n"
                       f"- **Minimum** : {min_val:.2f}%\n"
                       f"- **Maximum** : {max_val:.2f}%\n"
                       f"- **Nombre d'enregistrements** : {len(df):,}")
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions présentes dans les données."""
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"
                
                regions = df['region'].dropna().unique()
                
                if len(regions) == 0:
                    return "❌ Aucune région trouvée"
                
                regions_sorted = sorted(regions)
                result = f"🗺️ **{len(regions_sorted)} régions disponibles :**\n\n"
                
                for region in regions_sorted:
                    nb = len(df[df['region'] == region])
                    
                    if 'taux_regularite' in df.columns:
                        avg = df[df['region'] == region]['taux_regularite'].mean()
                        result += f"- **{region}** : {nb:,} enregistrements | Régularité : {avg:.2f}%\n"
                    else:
                        result += f"- **{region}** : {nb:,} enregistrements\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def top_regions_regulieres(n: int = 5) -> str:
            """
            Liste les N régions les plus régulières.
            
            Args:
                n: Nombre de régions (défaut 5)
            """
            try:
                if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Colonnes 'region' ou 'taux_regularite' manquantes"
                
                df_clean = df[['region', 'taux_regularite']].dropna()
                
                if len(df_clean) == 0:
                    return "❌ Aucune donnée valide"
                
                regularite_par_region = df_clean.groupby('region')['taux_regularite'].mean()
                n_regions = min(n, len(regularite_par_region))
                top = regularite_par_region.nlargest(n_regions)
                
                result = f"🏆 **Top {n_regions} régions les plus régulières :**\n\n"
                
                for i, (region, taux) in enumerate(top.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    result += f"{emoji} **{region}** : {taux:.2f}%\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def pires_regions(n: int = 5) -> str:
            """
            Liste les N régions avec la pire régularité.
            
            Args:
                n: Nombre de régions (défaut 5)
            """
            try:
                if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Colonnes 'region' ou 'taux_regularite' manquantes"
                
                df_clean = df[['region', 'taux_regularite']].dropna()
                
                if len(df_clean) == 0:
                    return "❌ Aucune donnée valide"
                
                regularite_par_region = df_clean.groupby('region')['taux_regularite'].mean()
                n_regions = min(n, len(regularite_par_region))
                bottom = regularite_par_region.nsmallest(n_regions)
                
                result = f"⚠️ **Top {n_regions} régions avec la pire régularité :**\n\n"
                
                for i, (region, taux) in enumerate(bottom.items(), 1):
                    result += f"{i}. **{region}** : {taux:.2f}% ⚠️\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def analyser_impact_meteo_global() -> str:
            """Analyse l'impact global de la météo sur la régularité."""
            try:
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
                
                result = "🌦️ **Impact météo sur la régularité :**\n\n"
                
                for cat in ['Bonne', 'Correcte', 'Difficile', 'Extrême']:
                    data = df_clean[df_clean['meteo_cat'] == cat]
                    if len(data) > 0:
                        avg = data['taux_regularite'].mean()
                        emoji = "☀️" if cat == 'Bonne' else "⛅" if cat == 'Correcte' else "🌧️" if cat == 'Difficile' else "⛈️"
                        result += f"{emoji} **Météo {cat}** : {avg:.2f}% ({len(data)} jours)\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # RETOUR DE TOUS LES OUTILS
        # ═══════════════════════════════════════════════════════════════
        
        return [
            # Debug
            debug_dataframe_info,
            
            # NOUVEAUX outils de filtrage
            filtrer_par_mois_annee_region,
            comparer_periodes,
            analyser_region_complete,
            chercher_anomalies_meteo,
            
            # Outils existants
            calculer_regularite_globale,
            liste_regions_disponibles,
            top_regions_regulieres,
            pires_regions,
            analyser_impact_meteo_global
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
        """Pose une question à l'agent"""
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            result = self.graph.invoke(initial_state)
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(final_message)
            
        except Exception as e:
            return f"❌ Erreur : {str(e)}"
    
    def reset_conversation(self):
        """Réinitialise l'historique (non implémenté avec LangGraph stateless)"""
        pass
