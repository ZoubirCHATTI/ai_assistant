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
        return "Données temporelles insuffisantes pour l'analyse"
    
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
    
    # ═══════════════════════════════════════════════════════════════════
    # OUTILS MÉTÉO (NOUVEAUX)
    # ═══════════════════════════════════════════════════════════════════
    
    @tool
    def verifier_donnees_meteo() -> str:
        """Vérifie si les données météo sont disponibles dans le dataset."""
        weather_cols = ['temperature_mean', 'precipitation', 'snow', 'wind_speed', 'weather_severity_score']
        
        available = [col for col in weather_cols if col in df.columns]
        missing = [col for col in weather_cols if col not in df.columns]
        
        if not available:
            return ("❌ Aucune donnée météo disponible.\n\n"
                   "Pour enrichir le dataset avec la météo :\n"
                   "1. Allez dans la page '🌦️ Analyse Météo'\n"
                   "2. Cliquez sur 'Lancer l'enrichissement météo'\n"
                   "3. Attendez quelques minutes\n"
                   "4. Revenez au Chat IA")
        
        result = f"✅ Données météo disponibles : {len(available)}/{len(weather_cols)} colonnes\n\n"
        result += f"📊 Colonnes présentes : {', '.join(available)}\n"
        
        if missing:
            result += f"\n⚠️ Colonnes manquantes : {', '.join(missing)}"
        
        # Statistiques rapides
        if 'weather_severity_score' in df.columns:
            avg_severity = df['weather_severity_score'].mean()
            result += f"\n\n🌦️ Score de sévérité moyen : {avg_severity:.1f}/100"
        
        return result
    
    @tool
    def analyser_impact_meteo_global() -> str:
        """Analyse l'impact global de la météo sur la régularité des trains."""
        
        # Vérifier la présence des colonnes nécessaires
        if 'weather_severity_score' not in df.columns:
            return ("❌ Données météo non disponibles.\n"
                   "Enrichissez d'abord le dataset dans la page '🌦️ Analyse Météo'")
        
        if 'taux_regularite' not in df.columns:
            return "❌ Colonne 'taux_regularite' non trouvée"
        
        # Créer des catégories météo
        df_clean = df.dropna(subset=['weather_severity_score', 'taux_regularite'])
        
        if len(df_clean) < 10:
            return "❌ Pas assez de données pour l'analyse"
        
        # Catégoriser
        conditions = [
            (df_clean['weather_severity_score'] <= 20, 'Bonne'),
            (df_clean['weather_severity_score'] <= 40, 'Correcte'),
            (df_clean['weather_severity_score'] <= 60, 'Difficile'),
            (df_clean['weather_severity_score'] > 60, 'Extrême')
        ]
        
        categories = []
        for condition, label in conditions:
            categories.append(label if condition.any() else None)
        
        df_clean['meteo_category'] = pd.cut(
            df_clean['weather_severity_score'],
            bins=[-1, 20, 40, 60, 100],
            labels=['Bonne', 'Correcte', 'Difficile', 'Extrême']
        )
        
        # Calculer les moyennes par catégorie
        regularite_by_meteo = df_clean.groupby('meteo_category')['taux_regularite'].agg(['mean', 'count'])
        
        result = "🌦️ **Impact de la météo sur la régularité des trains** :\n\n"
        
        for category in ['Bonne', 'Correcte', 'Difficile', 'Extrême']:
            if category in regularite_by_meteo.index:
                avg = regularite_by_meteo.loc[category, 'mean']
                count = int(regularite_by_meteo.loc[category, 'count'])
                
                emoji = "☀️" if category == 'Bonne' else "⛅" if category == 'Correcte' else "🌧️" if category == 'Difficile' else "⛈️"
                
                result += f"{emoji} **Météo {category}** ({count} jours) : {avg:.2f}% de régularité\n"
        
        # Calcul de la perte de régularité
        if 'Bonne' in regularite_by_meteo.index and 'Extrême' in regularite_by_meteo.index:
            perte = regularite_by_meteo.loc['Bonne', 'mean'] - regularite_by_meteo.loc['Extrême', 'mean']
            result += f"\n⚠️ **Perte de régularité** en météo extrême : **{perte:.2f} points**"
        
        return result
    
    @tool
    def impact_neige_sur_regularite() -> str:
        """Analyse l'impact spécifique de la neige sur la régularité."""
        
        if 'snow' not in df.columns:
            return "❌ Données de neige non disponibles. Enrichissez d'abord le dataset avec la météo."
        
        if 'taux_regularite' not in df.columns:
            return "❌ Colonne 'taux_regularite' non trouvée"
        
        df_clean = df.dropna(subset=['snow', 'taux_regularite'])
        
        if len(df_clean) < 5:
            return "❌ Pas assez de données pour l'analyse"
        
        # Séparer avec/sans neige
        df_avec_neige = df_clean[df_clean['snow'] > 0]
        df_sans_neige = df_clean[df_clean['snow'] == 0]
        
        if len(df_avec_neige) == 0:
            return "✅ Aucun épisode neigeux détecté dans la période analysée."
        
        regularite_avec = df_avec_neige['taux_regularite'].mean()
        regularite_sans = df_sans_neige['taux_regularite'].mean()
        difference = regularite_sans - regularite_avec
        
        nb_jours_neige = len(df_avec_neige)
        neige_moyenne = df_avec_neige['snow'].mean()
        neige_max = df_avec_neige['snow'].max()
        
        result = f"❄️ **Impact de la neige sur la régularité** :\n\n"
        result += f"📊 **Jours avec neige** : {nb_jours_neige} jours\n"
        result += f"❄️ **Neige moyenne** : {neige_moyenne:.1f} cm\n"
        result += f"❄️ **Neige maximale** : {neige_max:.1f} cm\n\n"
        result += f"✅ **Régularité sans neige** : {regularite_sans:.2f}%\n"
        result += f"❄️ **Régularité avec neige** : {regularite_avec:.2f}%\n\n"
        
        if difference > 5:
            result += f"⚠️ **Impact SIGNIFICATIF** : La neige réduit la régularité de **{difference:.2f} points** !"
        elif difference > 2:
            result += f"📊 **Impact modéré** : La neige réduit la régularité de {difference:.2f} points."
        else:
            result += f"✅ **Impact faible** : La neige a un effet limité ({difference:.2f} points)."
        
        return result
    
    @tool
    def impact_vent_fort() -> str:
        """Analyse l'impact des vents forts (>90 km/h) sur la régularité."""
        
        if 'wind_gusts' not in df.columns:
            return "❌ Données de vent non disponibles. Enrichissez d'abord le dataset avec la météo."
        
        if 'taux_regularite' not in df.columns:
            return "❌ Colonne 'taux_regularite' non trouvée"
        
        df_clean = df.dropna(subset=['wind_gusts', 'taux_regularite'])
        
        if len(df_clean) < 5:
            return "❌ Pas assez de données pour l'analyse"
        
        # Séparer vent fort / normal
        df_vent_fort = df_clean[df_clean['wind_gusts'] > 90]
        df_vent_normal = df_clean[df_clean['wind_gusts'] <= 90]
        
        if len(df_vent_fort) == 0:
            return "✅ Aucun épisode de vent fort (>90 km/h) détecté dans la période."
        
        regularite_vent_fort = df_vent_fort['taux_regularite'].mean()
        regularite_vent_normal = df_vent_normal['taux_regularite'].mean()
        difference = regularite_vent_normal - regularite_vent_fort
        
        nb_jours_vent_fort = len(df_vent_fort)
        vent_max = df_vent_fort['wind_gusts'].max()
        
        result = f"💨 **Impact des vents forts sur la régularité** :\n\n"
        result += f"📊 **Jours avec vent fort (>90 km/h)** : {nb_jours_vent_fort} jours\n"
        result += f"💨 **Rafale maximale enregistrée** : {vent_max:.0f} km/h\n\n"
        result += f"✅ **Régularité (vent normal)** : {regularite_vent_normal:.2f}%\n"
        result += f"💨 **Régularité (vent fort)** : {regularite_vent_fort:.2f}%\n\n"
        
        if difference > 5:
            result += f"⚠️ **Impact SIGNIFICATIF** : Le vent fort réduit la régularité de **{difference:.2f} points** !"
        elif difference > 2:
            result += f"📊 **Impact modéré** : Le vent fort réduit la régularité de {difference:.2f} points."
        else:
            result += f"✅ **Impact faible** : Le vent fort a un effet limité ({difference:.2f} points)."
        
        return result
    
    @tool
    def jours_meteo_extreme() -> str:
        """Liste les jours avec des conditions météo extrêmes."""
        
        if 'weather_severity_score' not in df.columns:
            return "❌ Score de sévérité météo non disponible. Enrichissez le dataset avec la météo."
        
        # Filtrer les jours extrêmes (score > 70)
        df_extreme = df[df['weather_severity_score'] > 70].copy()
        
        if len(df_extreme) == 0:
            return "✅ Aucune condition météo extrême détectée dans la période analysée."
        
        # Trier par sévérité
        df_extreme = df_extreme.sort_values('weather_severity_score', ascending=False)
        
        result = f"⛈️ **{len(df_extreme)} jours avec météo EXTRÊME** (score > 70) :\n\n"
        
        # Top 10 des pires jours
        for idx, row in df_extreme.head(10).iterrows():
            date = row.get('date', 'Date inconnue')
            region = row.get('region', 'Région inconnue')
            score = row['weather_severity_score']
            regularite = row.get('taux_regularite', None)
            
            date_str = date.strftime('%d/%m/%Y') if hasattr(date, 'strftime') else str(date)
            
            conditions = []
            if 'snow' in row and row['snow'] > 0:
                conditions.append(f"❄️ Neige: {row['snow']:.1f}cm")
            if 'wind_gusts' in row and row['wind_gusts'] > 90:
                conditions.append(f"💨 Vent: {row['wind_gusts']:.0f}km/h")
            if 'precipitation' in row and row['precipitation'] > 20:
                conditions.append(f"🌧️ Pluie: {row['precipitation']:.0f}mm")
            
            conditions_str = " | ".join(conditions) if conditions else "Conditions difficiles"
            
            result += f"📅 **{date_str}** - {region}\n"
            result += f"   Sévérité: {score:.0f}/100 | {conditions_str}\n"
            if regularite is not None:
                result += f"   Régularité: {regularite:.1f}%\n"
            result += "\n"
        
        return result
    
    return [
        calculer_regularite_globale,
        regularite_par_region,
        top_regions_regulieres,
        pires_regions,
        statistiques_trains,
        evolution_temporelle,
        comparer_periodes,
        liste_regions_disponibles,
        # Outils météo
        verifier_donnees_meteo,
        analyser_impact_meteo_global,
        impact_neige_sur_regularite,
        impact_vent_fort,
        jours_meteo_extreme
    ]
    
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
