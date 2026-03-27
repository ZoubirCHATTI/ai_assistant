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

from config import Config


class AgentState(TypedDict):
    """État de l'agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TERAnalysisAgent:
    """Agent IA pour analyser les données TER avec LangGraph"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df  # Stocker le DataFrame comme attribut de classe
        
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
        # OUTIL DE DEBUG (IMPORTANT)
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
                else:
                    info += f"❌ **Colonne 'region'** : NON trouvée\n"
                
                if 'taux_regularite' in df.columns:
                    avg = df['taux_regularite'].mean()
                    mini = df['taux_regularite'].min()
                    maxi = df['taux_regularite'].max()
                    info += f"✅ **Colonne 'taux_regularite'** : Moyenne {avg:.2f}% (min: {mini:.2f}%, max: {maxi:.2f}%)\n"
                else:
                    info += f"❌ **Colonne 'taux_regularite'** : NON trouvée\n"
                
                # Colonnes météo
                weather_cols = [col for col in df.columns if col in [
                    'temperature_mean', 'precipitation', 'snow', 'wind_speed', 
                    'wind_gusts', 'weather_severity_score'
                ]]
                
                if weather_cols:
                    info += f"\n🌦️ **Colonnes météo** ({len(weather_cols)}) : {', '.join(weather_cols)}\n"
                else:
                    info += f"\n❌ **Aucune colonne météo** trouvée\n"
                
                return info
                
            except Exception as e:
                return f"❌ Erreur debug : {str(e)}\n{type(e).__name__}"
        
        # ═══════════════════════════════════════════════════════════════
        # OUTILS D'ANALYSE STANDARD
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
        def statistiques_trains() -> str:
            """Donne des statistiques complètes sur le nombre de trains."""
            try:
                stats = []
                
                if 'nombre_trains_prevus' in df.columns:
                    total = df['nombre_trains_prevus'].sum()
                    stats.append(f"🚂 **Total trains prévus** : {total:,.0f}")
                
                if 'nombre_trains_circules' in df.columns:
                    total = df['nombre_trains_circules'].sum()
                    stats.append(f"✅ **Total trains circulés** : {total:,.0f}")
                
                if 'nombre_trains_supprimes' in df.columns:
                    total = df['nombre_trains_supprimes'].sum()
                    stats.append(f"❌ **Total trains supprimés** : {total:,.0f}")
                
                if 'nombre_trains_retard' in df.columns:
                    total = df['nombre_trains_retard'].sum()
                    stats.append(f"⏰ **Total trains en retard** : {total:,.0f}")
                
                return "\n".join(stats) if stats else "❌ Données de trains non disponibles"
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # OUTILS MÉTÉO
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def verifier_donnees_meteo() -> str:
            """Vérifie si les données météo sont disponibles."""
            try:
                weather_cols = ['temperature_mean', 'precipitation', 'snow', 'wind_speed', 'weather_severity_score']
                available = [col for col in weather_cols if col in df.columns]
                
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
        
        @tool
        def impact_neige_sur_regularite() -> str:
            """Analyse l'impact de la neige sur la régularité."""
            try:
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
                
                result = f"❄️ **Impact de la neige :**\n\n"
                result += f"- Sans neige : {reg_sans:.2f}%\n"
                result += f"- Avec neige : {reg_avec:.2f}%\n"
                result += f"- **Perte** : {diff:.2f} points\n"
                result += f"- Jours avec neige : {len(avec_neige)}"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def impact_vent_fort() -> str:
            """Analyse l'impact des vents forts (>90 km/h)."""
            try:
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
                
                result = f"💨 **Impact du vent fort :**\n\n"
                result += f"- Vent normal : {reg_normal:.2f}%\n"
                result += f"- Vent fort : {reg_fort:.2f}%\n"
                result += f"- **Perte** : {diff:.2f} points\n"
                result += f"- Jours avec vent fort : {len(vent_fort)}"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # RETOUR DE TOUS LES OUTILS
        # ═══════════════════════════════════════════════════════════════
        
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
    def calculate_regularite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le taux de régularité à partir des colonnes disponibles
        
        Args:
            df: DataFrame TER
            
        Returns:
            DataFrame avec la colonne taux_regularite ajoutée
        """
        df = df.copy()
        
        # Méthode 1 : Si on a "nombre_trains_a_l_heure"
        if 'nombre_trains_a_l_heure' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                df['nombre_trains_a_l_heure'] / df['nombre_trains_circules'] * 100
            ).fillna(0)
            print("✅ Taux de régularité calculé à partir de 'nombre_trains_a_l_heure'")
        
        # Méthode 2 : Si on a "nombre_trains_retard"
        elif 'nombre_trains_retard' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                100 - (df['nombre_trains_retard'] / df['nombre_trains_circules'] * 100)
            ).fillna(0)
            print("✅ Taux de régularité calculé à partir de 'nombre_trains_retard'")
        
        # Méthode 3 : Si on a un pourcentage de régularité direct
        elif 'regularite' in df.columns:
            df['taux_regularite'] = df['regularite']
            print("✅ Colonne 'regularite' renommée en 'taux_regularite'")
        
        # Méthode 4 : Chercher une colonne contenant "regularite" ou "ponctualite"
        else:
            regularite_cols = [col for col in df.columns if 'regularite' in col.lower() or 'ponctualite' in col.lower()]
            
            if regularite_cols:
                df['taux_regularite'] = df[regularite_cols[0]]
                print(f"✅ Colonne '{regularite_cols[0]}' utilisée comme taux_regularite")
            else:
                print("⚠️ Impossible de calculer le taux de régularité. Colonnes disponibles :")
                print(df.columns.tolist())
                # Valeur par défaut
                df['taux_regularite'] = 0
        
        # S'assurer que les valeurs sont entre 0 et 100
        df['taux_regularite'] = df['taux_regularite'].clip(0, 100)
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données TER depuis l'API
        
        Returns:
            DataFrame avec les données TER et le taux de régularité calculé
        """
        url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-ter/records"
        
        params = {
            'limit': 100,
            'offset': 0
        }
        
        all_records = []
        
        print("📥 Téléchargement des données TER...")
        
        while True:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                records = data.get('results', [])
                
                if not records:
                    break
                
                all_records.extend(records)
                print(f"  → {len(all_records)} enregistrements téléchargés...")
                
                if len(records) < params['limit']:
                    break
                
                params['offset'] += params['limit']
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Erreur lors du téléchargement : {e}")
                break
        
        if not all_records:
            raise ValueError("❌ Aucune donnée téléchargée")
        
        # Créer le DataFrame
        df = pd.DataFrame(all_records)
        
        print(f"✅ {len(df)} enregistrements chargés")
        print(f"📊 Colonnes disponibles : {', '.join(df.columns.tolist()[:10])}...")
        
        # Nettoyage de base
        df = self._clean_data(df)
        
        # NOUVEAU : Calculer le taux de régularité
        df = self.calculate_regularite(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et prépare les données"""
        df = df.copy()
        
        # Convertir les dates
        date_columns = ['date', 'mois', 'annee']
        for col in date_columns:
            if col in df.columns:
                if col == 'date':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convertir les colonnes numériques
        numeric_columns = [
            'nombre_trains_prevus', 'nombre_trains_circules',
            'nombre_trains_supprimes', 'nombre_trains_retard',
            'nombre_trains_a_l_heure'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Supprimer les lignes avec dates invalides
        if 'date' in df.columns:
            df = df.dropna(subset=['date'])
        
        return df        
