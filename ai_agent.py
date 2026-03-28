# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA conversationnel pour l'analyse des données TER.
Utilise Mistral AI (mistral-large-latest) via LangGraph avec :
- Outils structurés pour régularité, trains, météo
- Filtrage avancé par date / région / météo
- Historique de conversation multi-tours
- Prompt système riche et directif
"""

import re
import operator
import pandas as pd
from typing import TypedDict, Annotated, Sequence

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import Config


class AgentState(TypedDict):
    """État interne du graph LangGraph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TERAnalysisAgent:
    """
    Agent IA TER complet.
    - LangGraph + Mistral large pour le raisonnement
    - Outils structurés pour toutes les analyses (régularité, trains, météo)
    - Historique de conversation conservé entre les questions
    """

    MOIS_MAP = {
        'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4,
        'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8, 'aout': 8,
        'septembre': 9, 'octobre': 10, 'novembre': 11,
        'décembre': 12, 'decembre': 12
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.conversation_history: list[dict] = []

        # Conversion date si nécessaire
        if 'date' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        self.llm = ChatMistralAI(
            model=Config.MISTRAL_MODEL,
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0
        )

        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.system_prompt = self._build_system_prompt()
        self.graph = self._build_graph()

        print(f"✅ Agent IA initialisé — modèle : {Config.MISTRAL_MODEL} | {len(self.tools)} outils")

    def _build_system_prompt(self) -> str:
        """Construit un prompt système riche avec le contexte des données"""
        stats = f"Dataset TER : {len(self.df):,} enregistrements"

        if 'date' in self.df.columns:
            d_min = self.df['date'].min()
            d_max = self.df['date'].max()
            stats += f" | Période : {d_min.strftime('%Y-%m-%d')} → {d_max.strftime('%Y-%m-%d')}"

        if 'region' in self.df.columns:
            regions = sorted(self.df['region'].dropna().unique().tolist())
            stats += f" | {len(regions)} régions : {', '.join(regions[:8])}..."

        if 'taux_regularite' in self.df.columns:
            avg = self.df['taux_regularite'].mean()
            stats += f" | Régularité moyenne : {avg:.2f}%"

        weather_cols = [c for c in self.df.columns if c in [
            'temperature_mean', 'precipitation', 'snow', 'wind_speed',
            'wind_gusts', 'weather_severity_score'
        ]]
        meteo_info = f"Données météo disponibles : {', '.join(weather_cols)}" if weather_cols else \
                     "Données météo : non encore enrichies (page Analyse Météo)"

        return f"""Tu es un assistant expert en analyse de données ferroviaires TER en France.

**DONNÉES DISPONIBLES :**
{stats}
{meteo_info}

**RÈGLES ABSOLUES :**
1. TOUJOURS appeler un outil pour répondre à toute question portant sur des données chiffrées.
2. NE JAMAIS inventer de chiffres, de statistiques ou de noms de régions.
3. Répondre en français, de manière claire, concise et structurée.
4. Utiliser des émojis pour structurer (📊 📈 ❌ 🗺️ 🌦️ ❄️ 💨 🌧️).
5. Si une question est ambiguë, appeler d'abord `debug_dataframe_info` pour vérifier les colonnes disponibles.

**MAPPING MOIS :**
janvier=1 | février=2 | mars=3 | avril=4 | mai=5 | juin=6
juillet=7 | août=8 | septembre=9 | octobre=10 | novembre=11 | décembre=12

**OUTILS DISPONIBLES ET USAGE :**
- `debug_dataframe_info` → vérifie les colonnes et les données disponibles
- `calculer_regularite_globale` → statistiques globales de régularité
- `filtrer_par_date_et_region` → filtrage par mois, année et/ou région (outil principal)
- `analyser_region_complete` → analyse complète d'une région sur toute la période
- `comparer_deux_periodes` → compare deux périodes (ex: avril 2019 vs avril 2020)
- `liste_regions_disponibles` → liste toutes les régions avec leurs stats
- `top_regions_regulieres` → classement des N meilleures régions
- `pires_regions` → classement des N pires régions
- `statistiques_trains` → totaux de trains prévus, annulés, en retard
- `analyser_impact_meteo` → impact global de la météo sur la régularité
- `analyser_impact_neige` → impact spécifique de la neige
- `analyser_impact_vent` → impact des vents forts
- `analyser_impact_pluie` → impact des pluies fortes
- `correlation_meteo_regularite` → corrélation statistique météo/régularité

**EXEMPLES DE RAISONNEMENT :**
Question : "Combien de trains annulés en avril 2020 en Bretagne ?"
→ Appeler `filtrer_par_date_et_region(mois=4, annee=2020, region="Bretagne")`

Question : "Compare avril 2020 et avril 2019"
→ Appeler `comparer_deux_periodes(mois1=4, annee1=2020, mois2=4, annee2=2019)`

Question : "Impact de la neige ?"
→ Appeler `analyser_impact_neige()`

**NE RÉPONDS JAMAIS avec des chiffres sans avoir appelé un outil.**"""

    def _create_tools(self):
        """Crée tous les outils d'analyse disponibles pour l'agent"""
        df = self.df

        if df is None or len(df) == 0:
            raise ValueError("❌ Le DataFrame est vide ou None")

        print(f"🔧 Création des outils avec DataFrame de {len(df):,} lignes | colonnes : {list(df.columns)}")

        # ── Utilitaire interne ────────────────────────────────────────────

        def _find_annules_col(d):
            for c in ['nombre_trains_supprimes', 'nombre_trains_annules', 'nb_trains_annules']:
                if c in d.columns:
                    return c
            return None

        def _find_prevus_col(d):
            for c in ['nombre_trains_prevus', 'nb_trains_programmes']:
                if c in d.columns:
                    return c
            return None

        # ─────────────────────────────────────────────────────────────────
        # OUTILS DE BASE
        # ─────────────────────────────────────────────────────────────────

        @tool
        def debug_dataframe_info() -> str:
            """Affiche les informations de debug sur le DataFrame (colonnes, types, exemples)."""
            try:
                info = f"📊 **DataFrame TER — Debug**\n\n"
                info += f"- Lignes : {len(df):,}\n"
                info += f"- Colonnes ({len(df.columns)}) : {', '.join(df.columns.tolist())}\n\n"

                if 'region' in df.columns:
                    regions = sorted(df['region'].dropna().unique())
                    info += f"🗺️ Régions ({len(regions)}) : {', '.join(str(r) for r in regions)}\n\n"

                if 'date' in df.columns:
                    info += f"📅 Période : {df['date'].min()} → {df['date'].max()}\n"

                if 'taux_regularite' in df.columns:
                    avg = df['taux_regularite'].mean()
                    info += f"📈 Régularité moyenne : {avg:.2f}%\n"

                weather_cols = [c for c in df.columns if c in [
                    'temperature_mean', 'precipitation', 'snow',
                    'wind_speed', 'wind_gusts', 'weather_severity_score'
                ]]
                if weather_cols:
                    info += f"\n🌦️ Colonnes météo : {', '.join(weather_cols)}\n"
                else:
                    info += "\n❌ Pas de colonnes météo (données non enrichies)\n"

                return info
            except Exception as e:
                return f"❌ Erreur debug : {str(e)}"

        @tool
        def calculer_regularite_globale() -> str:
            """Calcule les statistiques globales de régularité sur toutes les données."""
            try:
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée"

                avg = df['taux_regularite'].mean()
                med = df['taux_regularite'].median()
                mini = df['taux_regularite'].min()
                maxi = df['taux_regularite'].max()

                return (
                    f"📊 **Régularité globale (toutes régions, toute la période) :**\n\n"
                    f"- Moyenne : **{avg:.2f}%**\n"
                    f"- Médiane : {med:.2f}%\n"
                    f"- Minimum : {mini:.2f}%\n"
                    f"- Maximum : {maxi:.2f}%\n"
                    f"- Enregistrements : {len(df):,}"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions avec leur nombre d'enregistrements et régularité moyenne."""
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"

                regions = sorted(df['region'].dropna().unique())
                if len(regions) == 0:
                    return "❌ Aucune région trouvée"

                result = f"🗺️ **{len(regions)} régions disponibles :**\n\n"
                for r in regions:
                    nb = len(df[df['region'] == r])
                    line = f"- **{r}** : {nb:,} enregistrements"
                    if 'taux_regularite' in df.columns:
                        avg = df[df['region'] == r]['taux_regularite'].mean()
                        line += f" | régularité moy. : {avg:.2f}%"
                    result += line + "\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def top_regions_regulieres(n: int = 5) -> str:
            """
            Classement des N régions les plus régulières.

            Args:
                n: Nombre de régions à afficher (défaut 5, max 20)
            """
            try:
                if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Colonnes 'region' ou 'taux_regularite' manquantes"

                n = min(n, df['region'].nunique())
                top = df.groupby('region')['taux_regularite'].mean().nlargest(n)

                result = f"🏆 **Top {n} régions les plus régulières :**\n\n"
                for i, (region, taux) in enumerate(top.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    result += f"{emoji} **{region}** : {taux:.2f}%\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def pires_regions(n: int = 5) -> str:
            """
            Classement des N régions avec la pire régularité.

            Args:
                n: Nombre de régions à afficher (défaut 5)
            """
            try:
                if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Colonnes 'region' ou 'taux_regularite' manquantes"

                n = min(n, df['region'].nunique())
                worst = df.groupby('region')['taux_regularite'].mean().nsmallest(n)

                result = f"📉 **{n} régions avec la pire régularité :**\n\n"
                for i, (region, taux) in enumerate(worst.items(), 1):
                    result += f"{i}. **{region}** : {taux:.2f}%\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        # ─────────────────────────────────────────────────────────────────
        # OUTIL PRINCIPAL DE FILTRAGE
        # ─────────────────────────────────────────────────────────────────

        @tool
        def filtrer_par_date_et_region(
            mois: int = 0,
            annee: int = 0,
            region: str = ""
        ) -> str:
            """
            Filtre les données TER par mois, année et/ou région, et retourne
            des statistiques complètes : régularité, trains annulés, trains prévus, météo.

            Args:
                mois: Numéro du mois (1=janvier ... 12=décembre). 0 = tous les mois.
                annee: Année (ex: 2020). 0 = toutes les années.
                region: Nom de la région (ex: "Bretagne"). Vide = toutes les régions.
            """
            try:
                if 'date' not in df.columns:
                    return "❌ Colonne 'date' non trouvée dans les données"

                d = df.copy()

                # Filtres date
                if annee and annee > 0:
                    d = d[d['date'].dt.year == annee]
                if mois and 1 <= mois <= 12:
                    d = d[d['date'].dt.month == mois]

                # Filtre région
                region_label = ""
                if region and region.strip() and 'region' in d.columns:
                    mask = d['region'].str.contains(region.strip(), case=False, na=False)
                    d = d[mask]
                    region_label = f" — {region.strip().capitalize()}"

                if len(d) == 0:
                    mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                                 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                    label_m = mois_noms[mois] if mois and 1 <= mois <= 12 else ""
                    label_a = str(annee) if annee else ""
                    return f"❌ Aucune donnée pour {label_m} {label_a}{region_label}"

                # Construction du titre
                mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                             'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                titre_mois = mois_noms[mois].capitalize() if mois and 1 <= mois <= 12 else "Toute la période"
                titre_annee = str(annee) if annee else ""
                result = f"📅 **{titre_mois} {titre_annee}{region_label}**\n\n"
                result += f"📊 {len(d):,} enregistrements\n\n"

                # Régularité
                if 'taux_regularite' in d.columns:
                    avg = d['taux_regularite'].mean()
                    result += f"📈 **Régularité moyenne** : {avg:.2f}%\n"

                # Trains annulés / supprimés
                col_ann = _find_annules_col(d)
                if col_ann:
                    total_ann = int(d[col_ann].sum())
                    result += f"❌ **Trains annulés/supprimés** : {total_ann:,}\n"

                # Trains prévus et taux d'annulation
                col_prev = _find_prevus_col(d)
                if col_prev:
                    total_prev = int(d[col_prev].sum())
                    result += f"🚆 **Trains prévus** : {total_prev:,}\n"
                    if col_ann and total_prev > 0:
                        taux_ann = total_ann / total_prev * 100
                        result += f"📊 **Taux d'annulation** : {taux_ann:.2f}%\n"

                # Trains en retard
                if 'nombre_trains_retard' in d.columns:
                    total_retard = int(d['nombre_trains_retard'].sum())
                    result += f"⏰ **Trains en retard** : {total_retard:,}\n"

                # Météo
                if 'weather_severity_score' in d.columns:
                    avg_meteo = d['weather_severity_score'].mean()
                    result += f"\n🌦️ **Score météo moyen** : {avg_meteo:.1f}/100"
                if 'snow' in d.columns:
                    jours_neige = (d['snow'] > 0).sum()
                    result += f"\n❄️ **Jours avec neige** : {jours_neige}"
                if 'wind_gusts' in d.columns:
                    max_vent = d['wind_gusts'].max()
                    result += f"\n💨 **Rafale max** : {max_vent:.0f} km/h"

                return result

            except Exception as e:
                import traceback
                return f"❌ Erreur filtrage : {str(e)}\n{traceback.format_exc()}"

        @tool
        def analyser_region_complete(region: str) -> str:
            """
            Analyse complète d'une région sur toute la période disponible.

            Args:
                region: Nom de la région (ex: "Bretagne", "Normandie")
            """
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"

                mask = df['region'].str.contains(region, case=False, na=False)
                d = df[mask]

                if len(d) == 0:
                    return (f"❌ Région '{region}' introuvable. "
                            f"Utilisez liste_regions_disponibles pour voir les noms exacts.")

                result = f"🗺️ **{region.capitalize()} — Analyse complète**\n\n"
                result += f"📊 {len(d):,} enregistrements\n"

                if 'date' in d.columns:
                    result += f"📅 Période : {d['date'].min().strftime('%d/%m/%Y')} → {d['date'].max().strftime('%d/%m/%Y')}\n"

                if 'taux_regularite' in d.columns:
                    avg = d['taux_regularite'].mean()
                    mini = d['taux_regularite'].min()
                    maxi = d['taux_regularite'].max()
                    result += f"\n📈 **Régularité :**\n"
                    result += f"- Moyenne : {avg:.2f}%\n"
                    result += f"- Minimum : {mini:.2f}%\n"
                    result += f"- Maximum : {maxi:.2f}%\n"

                col_ann = _find_annules_col(d)
                col_prev = _find_prevus_col(d)
                if col_ann and col_prev:
                    total_ann = int(d[col_ann].sum())
                    total_prev = int(d[col_prev].sum())
                    taux = total_ann / total_prev * 100 if total_prev > 0 else 0
                    result += f"\n❌ Trains annulés : {total_ann:,} / {total_prev:,} ({taux:.2f}%)\n"

                # Météo si disponible
                if 'weather_severity_score' in d.columns:
                    avg_meteo = d['weather_severity_score'].mean()
                    result += f"\n🌦️ Score météo moyen : {avg_meteo:.1f}/100"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def comparer_deux_periodes(
            mois1: int, annee1: int,
            mois2: int, annee2: int,
            region: str = ""
        ) -> str:
            """
            Compare deux périodes (mois/année) en termes de régularité et d'annulations.

            Args:
                mois1: Mois de la première période (1-12)
                annee1: Année de la première période
                mois2: Mois de la deuxième période (1-12)
                annee2: Année de la deuxième période
                region: Région (optionnel, vide = toutes)
            """
            try:
                if 'date' not in df.columns:
                    return "❌ Colonne 'date' manquante"

                mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                             'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']

                def get_period(mois, annee):
                    d = df[(df['date'].dt.month == mois) & (df['date'].dt.year == annee)].copy()
                    if region and region.strip() and 'region' in d.columns:
                        d = d[d['region'].str.contains(region, case=False, na=False)]
                    return d

                d1 = get_period(mois1, annee1)
                d2 = get_period(mois2, annee2)

                label1 = f"{mois_noms[mois1].capitalize()} {annee1}"
                label2 = f"{mois_noms[mois2].capitalize()} {annee2}"
                region_label = f" ({region.strip().capitalize()})" if region and region.strip() else ""

                result = f"📊 **Comparaison{region_label} : {label1} vs {label2}**\n\n"

                for label, d in [(label1, d1), (label2, d2)]:
                    result += f"**{label}** — {len(d):,} enregistrements\n"

                    if 'taux_regularite' in d.columns and len(d) > 0:
                        result += f"  📈 Régularité : {d['taux_regularite'].mean():.2f}%\n"

                    col_ann = _find_annules_col(d)
                    col_prev = _find_prevus_col(d)
                    if col_ann and len(d) > 0:
                        total_ann = int(d[col_ann].sum())
                        result += f"  ❌ Annulés : {total_ann:,}\n"
                        if col_prev:
                            total_prev = int(d[col_prev].sum())
                            taux = total_ann / total_prev * 100 if total_prev > 0 else 0
                            result += f"  📊 Taux annulation : {taux:.2f}%\n"
                    result += "\n"

                # Delta régularité
                if 'taux_regularite' in df.columns and len(d1) > 0 and len(d2) > 0:
                    diff = d1['taux_regularite'].mean() - d2['taux_regularite'].mean()
                    signe = "+" if diff > 0 else ""
                    result += f"📉 **Écart de régularité** : {signe}{diff:.2f} points ({label1} vs {label2})"

                return result
            except Exception as e:
                return f"❌ Erreur comparaison : {str(e)}"

        @tool
        def statistiques_trains() -> str:
            """Retourne les totaux globaux de trains prévus, annulés et en retard."""
            try:
                stats = []

                col_prev = _find_prevus_col(df)
                if col_prev:
                    stats.append(f"🚆 **Trains prévus** : {df[col_prev].sum():,.0f}")

                col_ann = _find_annules_col(df)
                if col_ann:
                    stats.append(f"❌ **Trains annulés** : {df[col_ann].sum():,.0f}")
                    if col_prev:
                        total_prev = df[col_prev].sum()
                        total_ann = df[col_ann].sum()
                        taux = total_ann / total_prev * 100 if total_prev > 0 else 0
                        stats.append(f"📊 **Taux annulation global** : {taux:.2f}%")

                if 'nombre_trains_retard' in df.columns:
                    stats.append(f"⏰ **Trains en retard** : {df['nombre_trains_retard'].sum():,.0f}")

                if 'nombre_trains_circules' in df.columns:
                    stats.append(f"✅ **Trains circulés** : {df['nombre_trains_circules'].sum():,.0f}")

                return "\n".join(stats) if stats else "❌ Aucune colonne de trains trouvée"
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        # ─────────────────────────────────────────────────────────────────
        # OUTILS MÉTÉO
        # ─────────────────────────────────────────────────────────────────

        @tool
        def analyser_impact_meteo() -> str:
            """Analyse l'impact global de la météo (score de sévérité) sur la régularité."""
            try:
                if 'weather_severity_score' not in df.columns or 'taux_regularite' not in df.columns:
                    return ("❌ Données météo non disponibles. "
                            "Allez dans la page 'Analyse Météo' pour enrichir le dataset.")

                d = df.dropna(subset=['weather_severity_score', 'taux_regularite']).copy()
                if len(d) < 10:
                    return "❌ Pas assez de données pour l'analyse"

                d['meteo_cat'] = pd.cut(
                    d['weather_severity_score'],
                    bins=[-1, 20, 40, 60, 100],
                    labels=['Bonne', 'Correcte', 'Difficile', 'Extrême']
                )

                result = "🌦️ **Impact météo sur la régularité :**\n\n"
                icons = {'Bonne': '☀️', 'Correcte': '⛅', 'Difficile': '🌧️', 'Extrême': '⛈️'}

                for cat in ['Bonne', 'Correcte', 'Difficile', 'Extrême']:
                    subset = d[d['meteo_cat'] == cat]
                    if len(subset) > 0:
                        avg = subset['taux_regularite'].mean()
                        result += f"{icons[cat]} **{cat}** : {avg:.2f}% ({len(subset):,} jours)\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def analyser_impact_neige() -> str:
            """Analyse l'impact de la neige sur la régularité des trains."""
            try:
                if 'snow' not in df.columns or 'taux_regularite' not in df.columns:
                    return ("❌ Données de neige non disponibles. "
                            "Enrichissez le dataset via la page 'Analyse Météo'.")

                d = df.dropna(subset=['snow', 'taux_regularite'])
                avec = d[d['snow'] > 0]
                sans = d[d['snow'] == 0]

                if len(avec) == 0:
                    return "✅ Aucun épisode neigeux détecté dans le dataset enrichi."

                reg_avec = avec['taux_regularite'].mean()
                reg_sans = sans['taux_regularite'].mean()
                diff = reg_sans - reg_avec

                return (
                    f"❄️ **Impact de la neige sur la régularité :**\n\n"
                    f"- Sans neige : {reg_sans:.2f}% ({len(sans):,} jours)\n"
                    f"- Avec neige : {reg_avec:.2f}% ({len(avec):,} jours)\n"
                    f"- **Perte de régularité** : {diff:.2f} points de pourcentage\n\n"
                    + ("⚠️ Impact significatif de la neige." if diff > 5 else
                       "📊 Impact modéré de la neige." if diff > 2 else
                       "✅ Impact faible de la neige.")
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def analyser_impact_vent() -> str:
            """Analyse l'impact des vents forts (rafales > 90 km/h) sur la régularité."""
            try:
                if 'wind_gusts' not in df.columns or 'taux_regularite' not in df.columns:
                    return ("❌ Données de vent non disponibles. "
                            "Enrichissez le dataset via la page 'Analyse Météo'.")

                d = df.dropna(subset=['wind_gusts', 'taux_regularite'])
                fort = d[d['wind_gusts'] > 90]
                normal = d[d['wind_gusts'] <= 90]

                if len(fort) == 0:
                    return "✅ Aucun vent fort (> 90 km/h) détecté dans le dataset enrichi."

                reg_fort = fort['taux_regularite'].mean()
                reg_normal = normal['taux_regularite'].mean()
                diff = reg_normal - reg_fort

                return (
                    f"💨 **Impact du vent fort (> 90 km/h) :**\n\n"
                    f"- Vent normal : {reg_normal:.2f}% ({len(normal):,} jours)\n"
                    f"- Vent fort : {reg_fort:.2f}% ({len(fort):,} jours)\n"
                    f"- **Perte de régularité** : {diff:.2f} points"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def analyser_impact_pluie() -> str:
            """Analyse l'impact des fortes pluies (> 10 mm) sur la régularité."""
            try:
                if 'rain' not in df.columns or 'taux_regularite' not in df.columns:
                    return ("❌ Données de pluie non disponibles. "
                            "Enrichissez le dataset via la page 'Analyse Météo'.")

                d = df.dropna(subset=['rain', 'taux_regularite'])
                forte = d[d['rain'] > 10]
                faible = d[d['rain'] <= 10]

                if len(forte) == 0:
                    return "✅ Aucun épisode de forte pluie (> 10 mm) détecté."

                reg_forte = forte['taux_regularite'].mean()
                reg_faible = faible['taux_regularite'].mean()
                diff = reg_faible - reg_forte

                return (
                    f"🌧️ **Impact de la pluie forte (> 10 mm) :**\n\n"
                    f"- Pluie faible : {reg_faible:.2f}% ({len(faible):,} jours)\n"
                    f"- Pluie forte : {reg_forte:.2f}% ({len(forte):,} jours)\n"
                    f"- **Perte de régularité** : {diff:.2f} points"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def correlation_meteo_regularite() -> str:
            """Calcule la corrélation statistique (Pearson) entre le score météo et la régularité."""
            try:
                if 'weather_severity_score' not in df.columns or 'taux_regularite' not in df.columns:
                    return ("❌ Données météo non disponibles. "
                            "Enrichissez le dataset via la page 'Analyse Météo'.")

                from scipy import stats as scipy_stats
                d = df.dropna(subset=['weather_severity_score', 'taux_regularite'])
                if len(d) < 10:
                    return "❌ Pas assez de données pour calculer une corrélation"

                r, p = scipy_stats.pearsonr(d['weather_severity_score'], d['taux_regularite'])
                abs_r = abs(r)

                if abs_r < 0.2:
                    force = "très faible"
                elif abs_r < 0.4:
                    force = "faible"
                elif abs_r < 0.6:
                    force = "modérée"
                elif abs_r < 0.8:
                    force = "forte"
                else:
                    force = "très forte"

                direction = "négative" if r < 0 else "positive"
                sig = "✅ Statistiquement significatif" if p < 0.05 else "⚠️ Non significatif"

                return (
                    f"📊 **Corrélation Météo ↔ Régularité :**\n\n"
                    f"- Coefficient de Pearson : **{r:.3f}**\n"
                    f"- Interprétation : corrélation {direction} {force}\n"
                    f"- P-value : {p:.4f} — {sig}\n"
                    f"- Données analysées : {len(d):,} enregistrements\n\n"
                    f"{'➡️ Plus la sévérité météo augmente, plus la régularité diminue.' if r < 0 else '➡️ Corrélation positive inattendue — vérifiez les données.'}"
                )
            except ImportError:
                return "❌ scipy non installé. Ajoutez scipy dans requirements.txt"
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        return [
            debug_dataframe_info,
            calculer_regularite_globale,
            filtrer_par_date_et_region,
            analyser_region_complete,
            comparer_deux_periodes,
            liste_regions_disponibles,
            top_regions_regulieres,
            pires_regions,
            statistiques_trains,
            analyser_impact_meteo,
            analyser_impact_neige,
            analyser_impact_vent,
            analyser_impact_pluie,
            correlation_meteo_regularite,
        ]

    def _build_graph(self):
        """Construit le graph LangGraph agent → tools → agent"""
        workflow = StateGraph(AgentState)

        def call_model(state: AgentState):
            full_messages = [SystemMessage(content=self.system_prompt)] + list(state["messages"])
            response = self.llm_with_tools.invoke(full_messages)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            last = state["messages"][-1]
            if hasattr(last, 'tool_calls') and last.tool_calls:
                return "tools"
            return END

        tool_node = ToolNode(self.tools)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def ask(self, question: str) -> str:
        """
        Pose une question à l'agent.
        L'historique des 6 derniers messages est injecté dans chaque appel.

        Args:
            question: Question en langage naturel

        Returns:
            Réponse textuelle de l'agent
        """
        if not Config.MISTRAL_API_KEY:
            return (
                "❌ **Clé API Mistral non configurée.**\n\n"
                "Ajoutez `MISTRAL_API_KEY` dans vos secrets Streamlit ou dans `.env`.\n"
                "Obtenez une clé gratuite sur https://console.mistral.ai/"
            )

        print(f"\n🤔 Question : {question}")

        # Construire les messages avec l'historique récent
        history_messages = []
        for msg in self.conversation_history[-6:]:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            # Les messages assistant sont déjà injectés via le graph

        initial_state = {
            "messages": history_messages + [HumanMessage(content=question)]
        }

        try:
            result = self.graph.invoke(initial_state)
            final = result["messages"][-1]
            response = final.content if hasattr(final, 'content') else str(final)

            # Mémoriser dans l'historique
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response})

            print(f"✅ Réponse générée ({len(response)} caractères)")
            return response

        except Exception as e:
            import traceback
            error = f"❌ **Erreur** : {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            return error

    def reset_conversation(self):
        """Réinitialise l'historique de conversation"""
        self.conversation_history = []
        print("🔄 Historique réinitialisé")

    def get_conversation_length(self) -> int:
        """Retourne le nombre de messages dans l'historique"""
        return len(self.conversation_history)
