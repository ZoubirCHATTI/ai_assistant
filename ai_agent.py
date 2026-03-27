# ai_agent.py
# -*- coding: utf-8 -*-
"""
Agent IA conversationnel pour l'analyse des données TER
Utilise Mistral AI via LangGraph avec outils structurés,
analyse directe des données et historique de conversation
"""

import re
import operator
import pandas as pd
from typing import TypedDict, Annotated, Sequence

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import Config


class AgentState(TypedDict):
    """État de l'agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TERAnalysisAgent:
    """
    Agent IA pour analyser les données TER.
    Combine LangGraph + outils Mistral + analyse directe + historique de conversation.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.conversation_history = []

        self.llm = ChatMistralAI(
            model=Config.MISTRAL_MODEL,
            mistral_api_key=Config.MISTRAL_API_KEY,
            temperature=0
        )

        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._create_graph()

        self.data_context = self._prepare_data_context()

        print(f"✅ Agent IA initialisé avec succès ({len(self.df):,} lignes, {len(self.tools)} outils)")

    def _prepare_data_context(self) -> str:
        """Prépare un résumé du contexte des données"""
        parts = [
            f"📊 Dataset TER : {len(self.df):,} enregistrements",
            f"📋 Colonnes disponibles : {', '.join(self.df.columns.tolist())}"
        ]

        if 'taux_regularite' in self.df.columns:
            avg = self.df['taux_regularite'].mean()
            mini = self.df['taux_regularite'].min()
            maxi = self.df['taux_regularite'].max()
            parts.append(f"📈 Régularité : moyenne={avg:.2f}%, min={mini:.2f}%, max={maxi:.2f}%")

        if 'region' in self.df.columns:
            nb = self.df['region'].nunique()
            exemples = sorted(self.df['region'].unique().tolist())[:10]
            parts.append(f"🗺️ Régions ({nb} au total) : {', '.join(exemples)}...")

        if 'date' in self.df.columns:
            date_min = self.df['date'].min()
            date_max = self.df['date'].max()
            parts.append(f"📅 Période : du {date_min} au {date_max}")

        return "\n".join(parts)

    def _analyze_data_for_question(self, question: str) -> str:
        """
        Analyse directement les données pour enrichir le contexte
        avant l'appel au LLM.
        """
        question_lower = question.lower()
        results = []

        try:
            df_filtered = self.df.copy()

            # Filtrage par mois
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

            year_match = re.search(r'\b(20\d{2})\b', question)
            if year_match:
                year = int(year_match.group(1))

            if 'date' in df_filtered.columns and (month or year):
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

                if year:
                    df_filtered = df_filtered[df_filtered['date'].dt.year == year]
                    results.append(f"📅 Filtré sur l'année {year}")

                if month:
                    df_filtered = df_filtered[df_filtered['date'].dt.month == month]
                    mois_nom_str = [k for k, v in mois_mapping.items() if v == month][0]
                    results.append(f"📅 Filtré sur le mois de {mois_nom_str}")

            # Filtrage par région
            if 'region' in df_filtered.columns:
                for region in df_filtered['region'].unique():
                    if region.lower() in question_lower:
                        df_filtered = df_filtered[df_filtered['region'] == region]
                        results.append(f"🗺️ Filtré sur la région : {region}")
                        break

            # Trains annulés
            if any(k in question_lower for k in ['annulé', 'annule', 'supprimé']):
                col = 'nb_trains_annules' if 'nb_trains_annules' in df_filtered.columns else \
                      'nb_train_annule' if 'nb_train_annule' in df_filtered.columns else None

                if col:
                    total = df_filtered[col].sum()
                    results.append(f"\n❌ **Trains annulés** : {int(total):,}")

                    if 'nb_trains_programmes' in df_filtered.columns:
                        total_prog = df_filtered['nb_trains_programmes'].sum()
                        taux = (total / total_prog * 100) if total_prog > 0 else 0
                        results.append(f"📊 Taux d'annulation : {taux:.2f}%")
                        results.append(f"🚆 Trains programmés : {int(total_prog):,}")

            # Régularité
            if any(k in question_lower for k in ['régularité', 'regularite', 'ponctualité']):
                if 'taux_regularite' in df_filtered.columns:
                    avg = df_filtered['taux_regularite'].mean()
                    mini = df_filtered['taux_regularite'].min()
                    maxi = df_filtered['taux_regularite'].max()
                    results.append(f"\n📈 **Régularité moyenne** : {avg:.2f}%")
                    results.append(f"📉 Régularité min : {mini:.2f}%")
                    results.append(f"📊 Régularité max : {maxi:.2f}%")

            # Top/pires régions
            if any(k in question_lower for k in ['meilleur', 'top']) and 'region' in df_filtered.columns:
                if 'taux_regularite' in df_filtered.columns:
                    top = df_filtered.groupby('region')['taux_regularite'].mean().nlargest(5)
                    results.append("\n🏆 **Top 5 meilleures régions** :")
                    for i, (r, s) in enumerate(top.items(), 1):
                        results.append(f"  {i}. {r} : {s:.2f}%")

            if any(k in question_lower for k in ['pire', 'worst']) and 'region' in df_filtered.columns:
                if 'taux_regularite' in df_filtered.columns:
                    worst = df_filtered.groupby('region')['taux_regularite'].mean().nsmallest(5)
                    results.append("\n📉 **5 pires régions** :")
                    for i, (r, s) in enumerate(worst.items(), 1):
                        results.append(f"  {i}. {r} : {s:.2f}%")

            # Résumé général
            if len(results) == 0 or any(k in question_lower for k in ['résumé', 'resume']):
                results.append(f"\n📊 **Enregistrements** : {len(df_filtered):,}")
                if 'taux_regularite' in df_filtered.columns:
                    results.append(f"📈 **Régularité moyenne** : {df_filtered['taux_regularite'].mean():.2f}%")
                col = 'nb_trains_annules' if 'nb_trains_annules' in df_filtered.columns else None
                if col:
                    results.append(f"❌ **Trains annulés** : {int(df_filtered[col].sum()):,}")

            return "\n".join(results) if results else "Aucune donnée correspondante trouvée."

        except Exception as e:
            return f"❌ Erreur lors de l'analyse directe : {str(e)}"

    def _create_tools(self):
        """Crée les outils d'analyse disponibles pour l'agent LangGraph"""

        df = self.df

        if df is None or len(df) == 0:
            raise ValueError("❌ Le DataFrame est vide ou None !")

        print(f"✅ Outils créés avec DataFrame de {len(df)} lignes et {len(df.columns)} colonnes")

        @tool
        def debug_dataframe_info() -> str:
            """Affiche des informations de debug sur le DataFrame."""
            try:
                info = f"📊 **Informations sur le DataFrame :**\n\n"
                info += f"- **Nombre de lignes** : {len(df):,}\n"
                info += f"- **Nombre de colonnes** : {len(df.columns)}\n"
                info += f"- **Colonnes** : {', '.join(df.columns.tolist())}\n\n"

                if 'region' in df.columns:
                    nb = df['region'].nunique()
                    exemples = df['region'].unique()[:5]
                    info += f"✅ **Colonne 'region'** : {nb} régions uniques\n"
                    info += f"   Exemples : {', '.join(str(r) for r in exemples)}\n"
                else:
                    info += "❌ **Colonne 'region'** : NON trouvée\n"

                if 'taux_regularite' in df.columns:
                    avg = df['taux_regularite'].mean()
                    mini = df['taux_regularite'].min()
                    maxi = df['taux_regularite'].max()
                    info += f"✅ **Colonne 'taux_regularite'** : Moyenne {avg:.2f}% (min: {mini:.2f}%, max: {maxi:.2f}%)\n"
                else:
                    info += "❌ **Colonne 'taux_regularite'** : NON trouvée\n"

                weather_cols = [c for c in df.columns if c in [
                    'temperature_mean', 'precipitation', 'snow',
                    'wind_speed', 'wind_gusts', 'weather_severity_score'
                ]]
                if weather_cols:
                    info += f"\n🌦️ **Colonnes météo** ({len(weather_cols)}) : {', '.join(weather_cols)}\n"
                else:
                    info += "\n❌ **Aucune colonne météo** trouvée\n"

                return info
            except Exception as e:
                return f"❌ Erreur debug : {str(e)}"

        @tool
        def calculer_regularite_globale() -> str:
            """Calcule le taux de régularité global sur toutes les données."""
            try:
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée"

                avg = df['taux_regularite'].mean()
                median = df['taux_regularite'].median()
                mini = df['taux_regularite'].min()
                maxi = df['taux_regularite'].max()

                return (
                    f"📊 **Statistiques de régularité globale :**\n\n"
                    f"- **Moyenne** : {avg:.2f}%\n"
                    f"- **Médiane** : {median:.2f}%\n"
                    f"- **Minimum** : {mini:.2f}%\n"
                    f"- **Maximum** : {maxi:.2f}%\n"
                    f"- **Enregistrements** : {len(df):,}"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions présentes dans les données."""
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"

                regions = sorted(df['region'].dropna().unique())

                if len(regions) == 0:
                    return "❌ Aucune région trouvée"

                result = f"🗺️ **{len(regions)} régions disponibles :**\n\n"
                for region in regions:
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

                top = df_clean.groupby('region')['taux_regularite'].mean().nlargest(min(n, df_clean['region'].nunique()))
                result = f"🏆 **Top {len(top)} régions les plus régulières :**\n\n"

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

                worst = df_clean.groupby('region')['taux_regularite'].mean().nsmallest(min(n, df_clean['region'].nunique()))
                result = f"📉 **{len(worst)} régions avec la pire régularité :**\n\n"

                for i, (region, taux) in enumerate(worst.items(), 1):
                    result += f"{i}. **{region}** : {taux:.2f}%\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def statistiques_trains() -> str:
            """Retourne les statistiques globales sur les trains programmés, annulés et en retard."""
            try:
                stats = []

                if 'nb_trains_programmes' in df.columns:
                    stats.append(f"🚆 **Total trains programmés** : {df['nb_trains_programmes'].sum():,.0f}")

                for col, label in [
                    ('nb_trains_annules', '❌ **Total trains annulés**'),
                    ('nb_train_annule', '❌ **Total trains annulés**'),
                    ('nombre_trains_retard', '⏰ **Total trains en retard**')
                ]:
                    if col in df.columns:
                        stats.append(f"{label} : {df[col].sum():,.0f}")

                return "\n".join(stats) if stats else "❌ Données de trains non disponibles"
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def verifier_donnees_meteo() -> str:
            """Vérifie si les données météo sont disponibles."""
            try:
                weather_cols = ['temperature_mean', 'precipitation', 'snow', 'wind_speed', 'weather_severity_score']
                available = [c for c in weather_cols if c in df.columns]

                if not available:
                    return "❌ Aucune donnée météo disponible"

                result = f"✅ **{len(available)}/{len(weather_cols)} colonnes météo disponibles**\n\n"
                result += f"**Colonnes** : {', '.join(available)}\n"

                if 'weather_severity_score' in df.columns:
                    avg = df['weather_severity_score'].mean()
                    result += f"\n🌦️ **Score sévérité moyen** : {avg:.1f}/100"

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

                df_clean = df_clean.copy()
                df_clean['meteo_cat'] = pd.cut(
                    df_clean['weather_severity_score'],
                    bins=[-1, 20, 40, 60, 100],
                    labels=['Bonne', 'Correcte', 'Difficile', 'Extrême']
                )

                result = "🌦️ **Impact météo sur la régularité :**\n\n"
                for cat, emoji in [('Bonne', '☀️'), ('Correcte', '⛅'), ('Difficile', '🌧️'), ('Extrême', '⛈️')]:
                    data = df_clean[df_clean['meteo_cat'] == cat]
                    if len(data) > 0:
                        avg = data['taux_regularite'].mean()
                        result += f"{emoji} **Météo {cat}** : {avg:.2f}% ({len(data)} jours)\n"

                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def impact_neige_sur_regularite() -> str:
            """Analyse l'impact de la neige sur la régularité."""
            try:
                if 'snow' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Données de neige non disponibles"

                df_clean = df.dropna(subset=['snow', 'taux_regularite'])
                avec = df_clean[df_clean['snow'] > 0]
                sans = df_clean[df_clean['snow'] == 0]

                if len(avec) == 0:
                    return "✅ Aucun épisode neigeux détecté"

                reg_avec = avec['taux_regularite'].mean()
                reg_sans = sans['taux_regularite'].mean()

                return (
                    f"❄️ **Impact de la neige :**\n\n"
                    f"- Sans neige : {reg_sans:.2f}%\n"
                    f"- Avec neige : {reg_avec:.2f}%\n"
                    f"- **Perte** : {reg_sans - reg_avec:.2f} points\n"
                    f"- Jours avec neige : {len(avec)}"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        @tool
        def impact_vent_fort() -> str:
            """Analyse l'impact des vents forts (>90 km/h)."""
            try:
                if 'wind_gusts' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Données de vent non disponibles"

                df_clean = df.dropna(subset=['wind_gusts', 'taux_regularite'])
                fort = df_clean[df_clean['wind_gusts'] > 90]
                normal = df_clean[df_clean['wind_gusts'] <= 90]

                if len(fort) == 0:
                    return "✅ Aucun vent fort (>90 km/h) détecté"

                reg_fort = fort['taux_regularite'].mean()
                reg_normal = normal['taux_regularite'].mean()

                return (
                    f"💨 **Impact du vent fort :**\n\n"
                    f"- Vent normal : {reg_normal:.2f}%\n"
                    f"- Vent fort : {reg_fort:.2f}%\n"
                    f"- **Perte** : {reg_normal - reg_fort:.2f} points\n"
                    f"- Jours avec vent fort : {len(fort)}"
                )
            except Exception as e:
                return f"❌ Erreur : {str(e)}"

        return [
            debug_dataframe_info,
            calculer_regularite_globale,
            liste_regions_disponibles,
            top_regions_regulieres,
            pires_regions,
            statistiques_trains,
            verifier_donnees_meteo,
            analyser_impact_meteo_global,
            impact_neige_sur_regularite,
            impact_vent_fort
        ]

    def _create_graph(self):
        """Crée le graph LangGraph"""

        workflow = StateGraph(AgentState)

        def call_model(state: AgentState):
            return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

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
        Enrichit le contexte avec une analyse directe des données
        avant d'invoquer le graph LangGraph.
        """
        print(f"\n🤔 Question reçue : {question}")

        if not Config.MISTRAL_API_KEY:
            return (
                "❌ **Erreur** : Clé API Mistral non configurée.\n\n"
                "Pour configurer :\n"
                "1. Créez un compte gratuit sur https://console.mistral.ai/\n"
                "2. Créez une clé API\n"
                "3. Ajoutez `MISTRAL_API_KEY=votre_clé` dans le fichier `.env`"
            )

        # Analyse directe pour enrichir le contexte
        data_analysis = self._analyze_data_for_question(question)
        print(f"📊 Analyse directe : {len(data_analysis)} caractères")

        # Construire le message enrichi
        enriched_question = (
            f"{question}\n\n"
            f"[Contexte des données]\n{self.data_context}\n\n"
            f"[Résultats de l'analyse directe]\n{data_analysis}"
        )

        # Ajouter à l'historique de conversation
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        try:
            # Garder les 6 derniers messages d'historique (3 échanges)
            history_messages = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else m["content"]
                for m in self.conversation_history[-6:-1]
            ]

            initial_state = {
                "messages": history_messages + [HumanMessage(content=enriched_question)]
            }

            result = self.graph.invoke(initial_state)
            final_message = result["messages"][-1]

            response = final_message.content if hasattr(final_message, 'content') else str(final_message)

            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            print("✅ Réponse générée")
            return response

        except Exception as e:
            error_msg = f"❌ **Erreur** : {str(e)}\n\n**Données analysées :**\n{data_analysis}"
            self.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

    def reset_conversation(self):
        """Réinitialise l'historique de conversation"""
        self.conversation_history = []
        print("🔄 Historique de conversation réinitialisé")

    def get_conversation_length(self) -> int:
        """Retourne le nombre de messages dans l'historique"""
        return len(self.conversation_history)
