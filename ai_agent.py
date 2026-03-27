# ai_agent.py

import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
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
        
        # Créer le prompt système
        self.system_prompt = self._create_system_prompt()
        
        # Créer le graph
        self.graph = self._create_graph()
        
        print(f"✅ Agent initialisé avec {len(self.tools)} outils disponibles")
    
    def _create_system_prompt(self) -> str:
        """Crée le prompt système directif"""
        
        # Statistiques du DataFrame
        stats = f"📊 Dataset : {len(self.df):,} enregistrements"
        
        if 'date' in self.df.columns:
            date_min = self.df['date'].min()
            date_max = self.df['date'].max()
            stats += f" | Période : {date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}"
        
        if 'region' in self.df.columns:
            nb_regions = self.df['region'].nunique()
            stats += f" | {nb_regions} régions"
        
        return f"""Tu es un assistant expert en analyse de données ferroviaires TER.

**DONNÉES DISPONIBLES :**
{stats}

**INSTRUCTIONS IMPORTANTES :**

1. **TOUJOURS utiliser les outils** pour répondre aux questions
2. **NE JAMAIS inventer** de chiffres ou de statistiques
3. Pour toute question nécessitant des données :
   - Identifie le bon outil à utiliser
   - Appelle-le avec les bons paramètres
   - Présente les résultats de manière claire

**OUTILS DISPONIBLES :**
- `filtrer_par_mois_annee_region` : Pour filtrer par date ET région (ex: avril 2020, Bretagne)
- `analyser_region_complete` : Pour analyser une région
- `comparer_periodes` : Pour comparer deux périodes
- `calculer_regularite_globale` : Pour les stats globales
- `liste_regions_disponibles` : Pour lister les régions
- `top_regions_regulieres` : Pour le top des régions
- `pires_regions` : Pour les pires régions
- `debug_dataframe_info` : Pour debugger

**MAPPING DES MOIS :**
janvier=1, février=2, mars=3, avril=4, mai=5, juin=6, juillet=7, août=8, septembre=9, octobre=10, novembre=11, décembre=12

**EXEMPLES :**

Question : "Combien de trains ont été annulés en avril 2020 en Bretagne ?"
→ Appelle `filtrer_par_mois_annee_region(mois=4, annee=2020, region="Bretagne")`

Question : "Quelle est la meilleure région ?"
→ Appelle `top_regions_regulieres(n=1)`

Question : "Compare avril 2020 et avril 2019 en Bretagne"
→ Appelle `comparer_periodes(mois1=4, annee1=2020, mois2=4, annee2=2019, region="Bretagne")`

**RÈGLES :**
- Si la question mentionne un mois ET une année : utilise `filtrer_par_mois_annee_region`
- Si la question demande une comparaison : utilise `comparer_periodes`
- Si la question porte sur une région : ajoute le paramètre `region`
- Réponds en français de manière concise et professionnelle

**NE RÉPONDS JAMAIS sans avoir appelé un outil si la question nécessite des données !**"""
    
    def _create_tools(self):
        """Crée les outils d'analyse disponibles pour l'agent"""
        
        # Référence locale EXPLICITE au DataFrame
        df = self.df
        
        # Vérification de sécurité
        if df is None or len(df) == 0:
            raise ValueError("❌ Le DataFrame est vide ou None !")
        
        print(f"🔧 Création des outils avec DataFrame de {len(df)} lignes")
        
        # ═══════════════════════════════════════════════════════════════
        # OUTIL DE DEBUG
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def debug_dataframe_info() -> str:
            """Affiche des informations de debug sur le DataFrame pour vérifier que les données sont accessibles."""
            try:
                info = f"📊 **DEBUG - Informations DataFrame :**\n\n"
                info += f"- Lignes : {len(df):,}\n"
                info += f"- Colonnes : {len(df.columns)}\n"
                info += f"- Noms : {', '.join(df.columns.tolist())}\n\n"
                
                if 'region' in df.columns:
                    regions = df['region'].unique()[:3]
                    info += f"- Régions (exemples) : {', '.join(str(r) for r in regions)}\n"
                
                if 'date' in df.columns:
                    info += f"- Période : {df['date'].min()} → {df['date'].max()}\n"
                
                return info
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # OUTIL PRINCIPAL DE FILTRAGE
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def filtrer_par_mois_annee_region(mois: int, annee: int, region: str = "") -> str:
            """
            Filtre les données TER par mois, année et optionnellement par région.
            Retourne des statistiques complètes sur la période.
            
            Args:
                mois: Numéro du mois (1=janvier, 2=février, ..., 12=décembre)
                annee: Année (ex: 2020)
                region: Nom de la région (optionnel, ex: "Bretagne"). Laisser vide pour toutes les régions.
            
            Returns:
                Statistiques formatées avec nombre de trains annulés, régularité, etc.
            """
            try:
                # Vérifier que la colonne date existe
                if 'date' not in df.columns:
                    return "❌ Colonne 'date' non trouvée dans les données"
                
                # Filtrer par année et mois
                mask_date = (df['date'].dt.year == annee) & (df['date'].dt.month == mois)
                df_filtered = df[mask_date].copy()
                
                if len(df_filtered) == 0:
                    mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                                'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                    return f"❌ Aucune donnée trouvée pour {mois_noms[mois]} {annee}"
                
                # Filtrer par région si spécifiée
                if region and region.strip() != "" and 'region' in df.columns:
                    mask_region = df_filtered['region'].str.contains(region, case=False, na=False)
                    df_filtered = df_filtered[mask_region]
                    
                    if len(df_filtered) == 0:
                        return f"❌ Aucune donnée pour la région '{region}' en {mois}/{annee}"
                
                # Construire le résultat
                mois_noms = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
                
                titre = f"📅 **{mois_noms[mois].capitalize()} {annee}"
                if region and region.strip():
                    titre += f" - {region.capitalize()}"
                titre += "**\n\n"
                
                result = titre
                result += f"📊 **{len(df_filtered):,} enregistrements**\n\n"
                
                # === RÉGULARITÉ ===
                if 'taux_regularite' in df_filtered.columns:
                    avg_reg = df_filtered['taux_regularite'].mean()
                    result += f"📈 **Régularité moyenne** : {avg_reg:.2f}%\n"
                
                # === TRAINS ANNULÉS ===
                col_annules = None
                possible_cols = ['nombre_trains_supprimes', 'nb_trains_annules', 'trains_annules']
                for col in possible_cols:
                    if col in df_filtered.columns:
                        col_annules = col
                        break
                
                if col_annules:
                    total_annules = int(df_filtered[col_annules].sum())
                    result += f"❌ **Trains annulés** : {total_annules:,}\n"
                else:
                    result += f"⚠️ Colonne trains annulés non trouvée\n"
                
                # === TRAINS PRÉVUS ===
                col_prevus = None
                possible_cols_prevus = ['nombre_trains_prevus', 'nb_trains_programmes', 'trains_prevus']
                for col in possible_cols_prevus:
                    if col in df_filtered.columns:
                        col_prevus = col
                        break
                
                if col_prevus:
                    total_prevus = int(df_filtered[col_prevus].sum())
                    result += f"🚆 **Trains prévus** : {total_prevus:,}\n"
                    
                    if col_annules and total_prevus > 0:
                        taux_annulation = (total_annules / total_prevus * 100)
                        result += f"📊 **Taux d'annulation** : {taux_annulation:.2f}%\n"
                
                # === MÉTÉO ===
                if 'weather_severity_score' in df_filtered.columns:
                    avg_meteo = df_filtered['weather_severity_score'].mean()
                    result += f"\n🌦️ **Score météo moyen** : {avg_meteo:.1f}/100"
                
                return result
                
            except Exception as e:
                import traceback
                return f"❌ Erreur lors du filtrage : {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        # ═══════════════════════════════════════════════════════════════
        # AUTRES OUTILS
        # ═══════════════════════════════════════════════════════════════
        
        @tool
        def analyser_region_complete(region: str) -> str:
            """
            Analyse complète d'une région spécifique sur toute la période disponible.
            
            Args:
                region: Nom de la région (ex: "Bretagne", "Normandie")
            """
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"
                
                mask = df['region'].str.contains(region, case=False, na=False)
                df_region = df[mask]
                
                if len(df_region) == 0:
                    return f"❌ Région '{region}' non trouvée. Utilisez l'outil liste_regions_disponibles pour voir les régions."
                
                result = f"🗺️ **{region.capitalize()} - Analyse complète**\n\n"
                result += f"📊 {len(df_region):,} enregistrements\n"
                
                if 'taux_regularite' in df_region.columns:
                    avg = df_region['taux_regularite'].mean()
                    mini = df_region['taux_regularite'].min()
                    maxi = df_region['taux_regularite'].max()
                    result += f"\n📈 **Régularité** :\n"
                    result += f"- Moyenne : {avg:.2f}%\n"
                    result += f"- Min : {mini:.2f}%\n"
                    result += f"- Max : {maxi:.2f}%\n"
                
                if 'date' in df_region.columns:
                    date_min = df_region['date'].min()
                    date_max = df_region['date'].max()
                    result += f"\n📅 Période : {date_min.strftime('%d/%m/%Y')} → {date_max.strftime('%d/%m/%Y')}"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def liste_regions_disponibles() -> str:
            """Liste toutes les régions présentes dans les données avec leurs statistiques."""
            try:
                if 'region' not in df.columns:
                    return "❌ Colonne 'region' non trouvée"
                
                regions = sorted(df['region'].dropna().unique())
                
                if len(regions) == 0:
                    return "❌ Aucune région trouvée"
                
                result = f"🗺️ **{len(regions)} régions disponibles :**\n\n"
                
                for region in regions:
                    nb = len(df[df['region'] == region])
                    result += f"- {region} ({nb:,} enregistrements)\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def top_regions_regulieres(n: int = 5) -> str:
            """
            Classement des N régions les plus régulières.
            
            Args:
                n: Nombre de régions à afficher (défaut: 5)
            """
            try:
                if 'region' not in df.columns or 'taux_regularite' not in df.columns:
                    return "❌ Colonnes nécessaires manquantes"
                
                reg_par_region = df.groupby('region')['taux_regularite'].mean().sort_values(ascending=False)
                top = reg_par_region.head(n)
                
                result = f"🏆 **Top {n} régions les plus régulières :**\n\n"
                
                for i, (region, taux) in enumerate(top.items(), 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    result += f"{emoji} {region} : {taux:.2f}%\n"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        @tool
        def calculer_regularite_globale() -> str:
            """Calcule les statistiques globales de régularité sur toutes les données."""
            try:
                if 'taux_regularite' not in df.columns:
                    return "❌ Colonne 'taux_regularite' non trouvée"
                
                avg = df['taux_regularite'].mean()
                median = df['taux_regularite'].median()
                mini = df['taux_regularite'].min()
                maxi = df['taux_regularite'].max()
                
                result = "📊 **Régularité globale (toutes régions, toute période) :**\n\n"
                result += f"- Moyenne : {avg:.2f}%\n"
                result += f"- Médiane : {median:.2f}%\n"
                result += f"- Min : {mini:.2f}%\n"
                result += f"- Max : {maxi:.2f}%\n"
                result += f"- Enregistrements : {len(df):,}"
                
                return result
            except Exception as e:
                return f"❌ Erreur : {str(e)}"
        
        # ═══════════════════════════════════════════════════════════════
        # RETOUR DES OUTILS
        # ═══════════════════════════════════════════════════════════════
        
        return [
            debug_dataframe_info,
            filtrer_par_mois_annee_region,
            analyser_region_complete,
            liste_regions_disponibles,
            top_regions_regulieres,
            calculer_regularite_globale
        ]
    
    def _create_graph(self):
        """Crée le graph LangGraph"""
        
        workflow = StateGraph(AgentState)
        
        def call_model(state: AgentState):
            """Appelle le modèle avec le prompt système"""
            messages = state["messages"]
            
            # Ajouter le prompt système au début
            full_messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = self.llm_with_tools.invoke(full_messages)
            return {"messages": [response]}
        
        tool_node = ToolNode(self.tools)
        
        def should_continue(state: AgentState):
            """Décide si on continue avec les outils ou on termine"""
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
            print(f"\n🤔 Question : {question}")
            
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            result = self.graph.invoke(initial_state)
            
            print(f"📨 Reçu {len(result['messages'])} messages")
            
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(final_message)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"❌ Erreur : {str(e)}\n\nDétails:\n{error_detail}"
    
    def reset_conversation(self):
        """Réinitialise l'historique"""
        pass
        
