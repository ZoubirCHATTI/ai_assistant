# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
🚆 ASSISTANT IA - ANALYSE TER SNCF
Application Streamlit pour l'analyse intelligente des données TER
═══════════════════════════════════════════════════════════════════════
"""

# app.py
# -*- coding: utf-8 -*-
"""
Dashboard d'analyse de la régularité des TER
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import Config
from data_loader import TERDataLoader
from weather_analyzer import WeatherAnalyzer
from ai_agent import TERAnalysisAgent

# ═══════════════════════════════════════════════════════════════════════
# FONCTION : GÉNÉRATION INTELLIGENTE DE GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════

def generate_smart_chart(question: str, df: pd.DataFrame):
    """
    Génère automatiquement un graphique intelligent selon la question
    
    Args:
        question: Question de l'utilisateur
        df: DataFrame avec les données
        
    Returns:
        Figure Plotly ou None
    """
    question_lower = question.lower()
    
    # Vérifier les colonnes essentielles
    if 'region' not in df.columns or 'taux_regularite' not in df.columns:
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # 1. GRAPHIQUES PAR RÉGION
    # ═══════════════════════════════════════════════════════════════
    
    if any(word in question_lower for word in ['région', 'region', 'par région']):
        # Calculer la moyenne par région
        region_stats = df.groupby('region')['taux_regularite'].mean().sort_values(ascending=False)
        
        # Filtrer selon le top/bottom demandé
        if 'top 5' in question_lower or '5 meilleur' in question_lower:
            region_stats = region_stats.head(5)
            title = "🏆 Top 5 des Régions - Régularité Moyenne"
        elif 'top 10' in question_lower or '10 meilleur' in question_lower:
            region_stats = region_stats.head(10)
            title = "🏆 Top 10 des Régions - Régularité Moyenne"
        elif '5 pire' in question_lower or 'bottom 5' in question_lower or 'worst 5' in question_lower:
            region_stats = region_stats.tail(5).sort_values(ascending=True)
            title = "📉 5 Pires Régions - Régularité Moyenne"
        elif '10 pire' in question_lower or 'bottom 10' in question_lower:
            region_stats = region_stats.tail(10).sort_values(ascending=True)
            title = "📉 10 Pires Régions - Régularité Moyenne"
        else:
            region_stats = region_stats.head(15)
            title = "📊 Régularité Moyenne par Région"
        
        # Type de graphique
        if 'camembert' in question_lower or 'pie' in question_lower:
            fig = px.pie(
                values=region_stats.values,
                names=region_stats.index,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            # Graphique en barres avec gradient de couleur
            fig = px.bar(
                x=region_stats.index,
                y=region_stats.values,
                title=title,
                labels={'x': 'Région', 'y': 'Taux de Régularité (%)'},
                color=region_stats.values,
                color_continuous_scale='RdYlGn',
                text=region_stats.values.round(2)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            fig.update_coloraxes(showscale=False)
        
        return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 2. ÉVOLUTION TEMPORELLE
    # ═══════════════════════════════════════════════════════════════
    
    elif any(word in question_lower for word in ['évolution', 'evolution', 'temps', 'tendance', 'courbe', 'mois']):
        if 'date' not in df.columns:
            return None
        
        # Évolution globale
        time_stats = df.groupby('date')['taux_regularite'].mean().reset_index()
        
        fig = px.line(
            time_stats,
            x='date',
            y='taux_regularite',
            title="📈 Évolution de la Régularité dans le Temps",
            labels={'date': 'Date', 'taux_regularite': 'Taux de Régularité (%)'},
            markers=True
        )
        
        fig.update_traces(
            line_color='#1f77b4',
            line_width=3,
            marker=dict(size=6)
        )
        
        fig.update_layout(
            hovermode='x unified',
            yaxis_range=[time_stats['taux_regularite'].min() - 5, 100]
        )
        
        # Ajouter une ligne de tendance
        fig.add_scatter(
            x=time_stats['date'],
            y=time_stats['taux_regularite'].rolling(7).mean(),
            mode='lines',
            name='Tendance (7 jours)',
            line=dict(color='red', width=2, dash='dash')
        )
        
        return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 3. COMPARAISON DE RÉGIONS SPÉCIFIQUES
    # ═══════════════════════════════════════════════════════════════
    
    elif 'compare' in question_lower or 'comparaison' in question_lower:
        # Extraire les noms de régions mentionnés
        regions_mentioned = [region for region in df['region'].unique() if region.lower() in question_lower]
        
        if len(regions_mentioned) >= 2 and 'date' in df.columns:
            # Comparaison temporelle de plusieurs régions
            df_compare = df[df['region'].isin(regions_mentioned)]
            region_time = df_compare.groupby(['date', 'region'])['taux_regularite'].mean().reset_index()
            
            fig = px.line(
                region_time,
                x='date',
                y='taux_regularite',
                color='region',
                title=f"📊 Comparaison : {' vs '.join(regions_mentioned)}",
                labels={'date': 'Date', 'taux_regularite': 'Taux de Régularité (%)', 'region': 'Région'},
                markers=True
            )
            
            fig.update_layout(hovermode='x unified')
            return fig
        
        elif len(regions_mentioned) >= 2:
            # Comparaison simple en barres
            region_stats = df[df['region'].isin(regions_mentioned)].groupby('region')['taux_regularite'].mean()
            
            fig = px.bar(
                x=region_stats.index,
                y=region_stats.values,
                title=f"📊 Comparaison : {' vs '.join(regions_mentioned)}",
                labels={'x': 'Région', 'y': 'Taux de Régularité (%)'},
                color=region_stats.values,
                color_continuous_scale='RdYlGn',
                text=region_stats.values.round(2)
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_coloraxes(showscale=False)
            return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 4. HISTOGRAMME / DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════
    
    elif any(word in question_lower for word in ['distribution', 'histogramme', 'histogram', 'répartition']):
        fig = px.histogram(
            df,
            x='taux_regularite',
            nbins=40,
            title="📊 Distribution des Taux de Régularité",
            labels={'taux_regularite': 'Taux de Régularité (%)', 'count': 'Nombre d\'enregistrements'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Ajouter une ligne verticale pour la moyenne
        mean_val = df['taux_regularite'].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Moyenne: {mean_val:.1f}%",
            annotation_position="top"
        )
        
        return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 5. BOX PLOT PAR RÉGION
    # ═══════════════════════════════════════════════════════════════
    
    elif 'box' in question_lower or 'boxplot' in question_lower:
        # Top 10 régions pour la lisibilité
        top_regions = df.groupby('region')['taux_regularite'].mean().nlargest(10).index
        df_box = df[df['region'].isin(top_regions)]
        
        fig = px.box(
            df_box,
            x='region',
            y='taux_regularite',
            title="📦 Distribution de la Régularité par Région (Top 10)",
            labels={'region': 'Région', 'taux_regularite': 'Taux de Régularité (%)'},
            color='region'
        )
        
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 6. SCATTER PLOT
    # ═══════════════════════════════════════════════════════════════
    
    elif 'scatter' in question_lower or 'nuage' in question_lower:
        if 'nb_trains_programmes' in df.columns:
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x='nb_trains_programmes',
                y='taux_regularite',
                color='region',
                title="🎯 Régularité vs Nombre de Trains Programmés",
                labels={
                    'nb_trains_programmes': 'Nombre de trains programmés',
                    'taux_regularite': 'Taux de Régularité (%)'
                },
                opacity=0.6
            )
            
            return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 7. ANALYSE MÉTÉO (si données enrichies)
    # ═══════════════════════════════════════════════════════════════
    
    elif any(word in question_lower for word in ['météo', 'meteo', 'neige', 'pluie', 'vent', 'température']):
        if 'weather_snowfall' in df.columns:
            # Catégoriser la neige
            df_weather = df.copy()
            df_weather['snow_category'] = pd.cut(
                df_weather['weather_snowfall'],
                bins=[-0.1, 0, 5, 20, 1000],
                labels=['Pas de neige', 'Neige légère', 'Neige modérée', 'Forte neige']
            )
            
            snow_stats = df_weather.groupby('snow_category')['taux_regularite'].mean()
            
            fig = px.bar(
                x=snow_stats.index,
                y=snow_stats.values,
                title="🌨️ Impact de la Neige sur la Régularité",
                labels={'x': 'Condition de neige', 'y': 'Taux de Régularité (%)'},
                color=snow_stats.values,
                color_continuous_scale='Blues',
                text=snow_stats.values.round(2)
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            return fig
        
        elif 'weather_temperature' in df.columns:
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x='weather_temperature',
                y='taux_regularite',
                color='region',
                title="🌡️ Impact de la Température sur la Régularité",
                labels={
                    'weather_temperature': 'Température (°C)',
                    'taux_regularite': 'Taux de Régularité (%)'
                },
                opacity=0.5,
                trendline="lowess"
            )
            
            return fig
    
    # ═══════════════════════════════════════════════════════════════
    # 8. GRAPHIQUE PAR DÉFAUT : TOP 10 RÉGIONS
    # ═══════════════════════════════════════════════════════════════
    
    else:
        region_stats = df.groupby('region')['taux_regularite'].mean().nlargest(10)
        
        fig = px.bar(
            x=region_stats.index,
            y=region_stats.values,
            title="🏆 Top 10 Régions - Régularité Moyenne",
            labels={'x': 'Région', 'y': 'Taux de Régularité (%)'},
            color=region_stats.values,
            color_continuous_scale='RdYlGn',
            text=region_stats.values.round(2)
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        fig.update_coloraxes(showscale=False)
        
        return fig

st.set_page_config(
    page_title="TER Analysis Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'df_enriched' not in st.session_state:
    st.session_state.df_enriched = None

if 'current_df_hash' not in st.session_state:
    st.session_state.current_df_hash = None


@st.cache_data(ttl=3600, show_spinner=False)
def load_ter_data():
    """Charge les données TER avec cache (1 heure)"""
    try:
        loader = TERDataLoader()
        df = loader.load_data()
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


with st.spinner("⏳ Chargement des données TER depuis l'API SNCF..."):
    df = load_ter_data()

if df is None or len(df) == 0:
    st.error("❌ Impossible de charger les données TER")
    st.info("""
    **Causes possibles :**
    - L'API SNCF est temporairement indisponible
    - Problème de connexion internet
    - Le dataset n'existe plus ou a changé d'URL
    
    **Solution :**
    - Vérifiez votre connexion internet
    - Réessayez dans quelques minutes
    - Contactez le support si le problème persiste
    """)
    st.stop()

st.success(f"✅ {len(df):,} enregistrements chargés depuis l'API SNCF")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'taux_regularite' in df.columns:
        avg_reg = df['taux_regularite'].mean()
        st.metric("📊 Régularité moyenne", f"{avg_reg:.2f}%")

with col2:
    if 'region' in df.columns:
        nb_regions = df['region'].nunique()
        st.metric("🗺️ Régions", nb_regions)

with col3:
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        nb_mois = (date_max.year - date_min.year) * 12 + (date_max.month - date_min.month) + 1
        st.metric("📅 Période", f"{nb_mois} mois")

with col4:
    st.metric("📦 Enregistrements", f"{len(df):,}")

st.markdown("---")

weather_analyzer = WeatherAnalyzer(df)

st.sidebar.title("🚆 TER Analysis Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Vue d'ensemble", "🌦️ Analyse Météo", "💬 Chat IA"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**À propos**

Dashboard d'analyse de la régularité des trains TER en France.

**Données :** API SNCF Open Data  
**IA :** Mistral AI  
**Météo :** Open-Meteo
""")

if page == "🏠 Accueil":
    st.title("🚆 Dashboard d'Analyse TER")

    st.markdown("""
    ## Bienvenue sur le Dashboard d'Analyse de la Régularité des TER

    Ce tableau de bord vous permet d'analyser la ponctualité des trains TER en France.

    ### 📊 Fonctionnalités disponibles :

    #### 🏠 **Accueil**
    - Vue d'ensemble du projet
    - Informations sur les données

    #### 📊 **Vue d'ensemble**
    - Statistiques générales de régularité
    - Comparaison par région
    - Évolution temporelle
    - Visualisations interactives

    #### 🌦️ **Analyse Météo**
    - Enrichissement des données avec la météo
    - Impact de la neige, du vent, des précipitations
    - Corrélation météo-régularité

    #### 💬 **Chat IA**
    - Posez vos questions en langage naturel
    - Analyses personnalisées
    - Réponses basées sur les données réelles

    ### 🚀 Pour commencer :

    1. Explorez la **Vue d'ensemble** pour voir les statistiques globales
    2. Enrichissez avec la **Météo** pour des analyses approfondies
    3. Utilisez le **Chat IA** pour des questions spécifiques
    """)

    st.markdown("---")

    st.subheader("📈 Aperçu rapide des données")

    if 'taux_regularite' in df.columns and 'region' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            top5 = df.groupby('region')['taux_regularite'].mean().nlargest(5)
            st.write("**🏆 Top 5 régions les plus régulières**")
            for i, (region, taux) in enumerate(top5.items(), 1):
                st.write(f"{i}. {region}: {taux:.2f}%")

        with col2:
            bottom5 = df.groupby('region')['taux_regularite'].mean().nsmallest(5)
            st.write("**⚠️ Top 5 régions les moins régulières**")
            for i, (region, taux) in enumerate(bottom5.items(), 1):
                st.write(f"{i}. {region}: {taux:.2f}%")
# ═══════════════════════════════════════════════════════════════════════
# PAGE : 📊 DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.title("📊 Dashboard de Ponctualité TER")
    
    # Filtres globaux
    with st.expander("🔧 Filtres", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'region' in df.columns:
                regions = ['Toutes'] + sorted(df['region'].unique().tolist())
                selected_region = st.selectbox("Région", regions)
            else:
                selected_region = 'Toutes'
        
        with col2:
            if 'annee' in df.columns:
                annees = ['Toutes'] + sorted(df['annee'].unique().tolist(), reverse=True)
                selected_annee = st.selectbox("Année", annees)
            else:
                selected_annee = 'Toutes'
        
        with col3:
            if 'mois' in df.columns:
                mois = ['Tous'] + sorted(df['mois'].unique().tolist())
                selected_mois = st.selectbox("Mois", mois)
            else:
                selected_mois = 'Tous'
    
    # Application des filtres
    df_filtered = df.copy()
    
    if selected_region != 'Toutes' and 'region' in df.columns:
        df_filtered = df_filtered[df_filtered['region'] == selected_region]
    
    if selected_annee != 'Toutes' and 'annee' in df.columns:
        df_filtered = df_filtered[df_filtered['annee'] == selected_annee]
    
    if selected_mois != 'Tous' and 'mois' in df.columns:
        df_filtered = df_filtered[df_filtered['mois'] == selected_mois]
    
    # Affichage des KPIs
    plot_kpi_cards(df_filtered)
    
    st.markdown("---")
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        plot_regularite_evolution(df_filtered)
    
    with col2:
        plot_regularite_by_region(df_filtered)
    
    st.markdown("---")
    
    # Analyse des perturbations
    plot_causes_retards(df_filtered)
    
    st.markdown("---")
    
    # Heatmap
    plot_heatmap_regularite(df_filtered)

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 💬 CHAT IA
# ═══════════════════════════════════════════════════════════════════════

elif page == "💬 Chat IA":
    st.title("💬 Chat avec l'Assistant IA")

    st.markdown("""
    Posez vos questions en français sur les données TER. L'IA peut analyser les données **ET générer des graphiques** automatiquement !
    """)

    if not Config.MISTRAL_API_KEY:
        st.error("❌ **Clé API Mistral non configurée**")

        st.markdown("""
        ### 🔑 Comment obtenir une clé API Mistral (gratuite) ?
        
        1. **Créez un compte** sur https://console.mistral.ai/
        2. **Connectez-vous** et allez dans **"API Keys"**
        3. Cliquez sur **"Create API Key"**
        4. **Copiez** la clé générée
        
        ### ⚙️ Configuration
        
        **En local :**
```bash
        # Créez un fichier .env
        MISTRAL_API_KEY=votre_clé_ici
```
        
        **Sur Streamlit Cloud :**
        - Settings → Secrets
        - Ajoutez : `MISTRAL_API_KEY = "votre_clé_ici"`
        """)

        st.stop()

    agent_df = st.session_state.df_enriched if st.session_state.df_enriched is not None else df
    df_hash = hash(str(agent_df.shape) + str(agent_df.columns.tolist()))

    if (st.session_state.agent is None or st.session_state.current_df_hash != df_hash):
        try:
            with st.spinner("🤖 Initialisation de l'agent IA..."):
                st.session_state.agent = TERAnalysisAgent(agent_df)
                st.session_state.current_df_hash = df_hash
            st.success("✅ Agent IA prêt !", icon="🤖")
        except Exception as e:
            st.error(f"❌ **Erreur lors de l'initialisation** : {str(e)}")
            st.stop()

    col_header1, col_header2 = st.columns([4, 1])
    with col_header2:
        if st.button("🔄 Recharger", help="Réinitialiser l'agent"):
            st.session_state.agent = None
            st.session_state.current_df_hash = None
            st.rerun()

    with st.expander("💡 Exemples de questions avec graphiques", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📊 Graphiques par région :**
            - Montre-moi un graphique des régions
            - Compare les 5 meilleures régions en barres
            - Fais un camembert de la régularité par région
            - Trace les 10 pires régions
            
            **📈 Évolutions temporelles :**
            - Trace l'évolution de la régularité
            - Montre l'évolution mois par mois
            - Graphique de la tendance sur l'année
            """)

        with col2:
            st.markdown("""
            **📉 Analyses avancées :**
            - Histogramme de la régularité
            - Box plot par région
            - Scatter plot régularité vs trains
            - Heatmap des corrélations
            
            **🌦️ Impact météo (si enrichi) :**
            - Impact de la neige en graphique
            - Corrélation température et régularité
            """)

    st.markdown("---")

    if not st.session_state.chat_history:
        st.info("👋 **Bonjour !** Demandez-moi une analyse ou un graphique sur les données TER.", icon="🤖")
    else:
        for idx, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message['content'])

                    if 'figure' in message and message['figure'] is not None:
                        st.plotly_chart(message['figure'], use_container_width=True, key=f"chart_{idx}")

    user_question = st.chat_input("💬 Posez votre question ici...")

    if user_question:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Analyse et génération..."):
                try:
                    response = st.session_state.agent.ask(user_question)
                    st.markdown(response)

                    plot_keywords = [
                        'graphique', 'graph', 'courbe', 'trace', 'dessine', 'montre',
                        'visualise', 'affiche', 'camembert', 'histogramme', 'barres',
                        'plot', 'chart', 'diagramme', 'évolution', 'compare', 'comparaison',
                        'heatmap', 'scatter', 'box plot', 'pie', 'tendance'
                    ]

                    should_plot = any(keyword in user_question.lower() for keyword in plot_keywords)

                    response_message = {
                        'role': 'assistant',
                        'content': response,
                        'figure': None
                    }

                    if should_plot:
                        try:
                            fig = generate_smart_chart(user_question, agent_df)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                response_message['figure'] = fig
                        except Exception as plot_error:
                            st.warning(f"⚠️ Graphique non généré : {str(plot_error)}")

                    st.session_state.chat_history.append(response_message)

                except Exception as e:
                    error_msg = f"❌ **Erreur** : {str(e)}"
                    st.error(error_msg)

                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': error_msg,
                        'figure': None
                    })

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🗑️ Effacer historique"):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()

    with col2:
        nb_questions = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        nb_charts = len([m for m in st.session_state.chat_history if m.get('figure') is not None])
        st.metric("💬 Questions", nb_questions)
        st.metric("📊 Graphiques", nb_charts)

    with col3:
        if st.session_state.df_enriched is not None:
            st.success("✅ Données enrichies disponibles")
        else:
            st.info("ℹ️ Enrichissez avec 'Analyse Météo'")
# ═══════════════════════════════════════════════════════════════════════
# PAGE : 📈 VISUALISATIONS PERSONNALISÉES
# ═══════════════════════════════════════════════════════════════════════

elif page == "📈 Visualisations Personnalisées":
    st.title("📈 Créateur de Visualisations")
    
    st.markdown("Créez vos propres graphiques en sélectionnant les paramètres ci-dessous.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Type de graphique
        chart_type = st.selectbox(
            "Type de graphique",
            ["Ligne", "Barre", "Scatter", "Histogramme", "Box Plot"]
        )
        
        # Colonnes disponibles
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Sélection des axes
        x_col = st.selectbox("Axe X", all_cols)
        
        if chart_type != "Histogramme":
            y_col = st.selectbox("Axe Y", numeric_cols)
        else:
            y_col = None
        
        # Couleur (optionnel)
        use_color = st.checkbox("Ajouter une dimension de couleur")
        if use_color:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            color_col = st.selectbox("Colonne de couleur", categorical_cols)
        else:
            color_col = None
        
        # Filtres
        st.markdown("---")
        st.subheader("🔍 Filtres")
        
        if 'region' in df.columns:
            regions = ['Toutes'] + sorted(df['region'].unique().tolist())
            filter_region = st.multiselect("Régions", regions, default=['Toutes'])
        else:
            filter_region = ['Toutes']
        
        # Bouton de génération
        generate_viz = st.button("🎨 Générer le graphique", use_container_width=True)
    
    with col2:
        st.subheader("📊 Résultat")
        
        if generate_viz:
            # Appliquer les filtres
            df_viz = df.copy()
            
            if 'Toutes' not in filter_region and 'region' in df.columns:
                df_viz = df_viz[df_viz['region'].isin(filter_region)]
            
            # Créer la visualisation
            fig = plot_custom_visualization(df_viz, chart_type, x_col, y_col, color_col)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le graphique (HTML)",
                    data=fig.to_html(),
                    file_name=f"chart_{chart_type.lower()}_{x_col}.html",
                    mime="text/html"
                )
            else:
                st.warning("⚠️ Impossible de créer le graphique avec ces paramètres.")
        else:
            st.info("👈 Configurez votre graphique et cliquez sur 'Générer'")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🔍 EXPLORATEUR DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔍 Explorateur de Données":
    st.title("🔍 Explorateur de Données")
    
    # Filtres avancés
    with st.expander("🔧 Filtres Avancés", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'region' in df.columns:
                regions_filter = st.multiselect(
                    "Régions",
                    options=sorted(df['region'].unique()),
                    default=[]
                )
            else:
                regions_filter = []
        
        with col2:
            if 'annee' in df.columns:
                annees_filter = st.multiselect(
                    "Années",
                    options=sorted(df['annee'].unique(), reverse=True),
                    default=[]
                )
            else:
                annees_filter = []
        
        with col3:
            if 'taux_regularite' in df.columns:
                min_reg, max_reg = float(df['taux_regularite'].min()), float(df['taux_regularite'].max())
                regularite_range = st.slider(
                    "Plage de régularité (%)",
                    min_value=min_reg,
                    max_value=max_reg,
                    value=(min_reg, max_reg)
                )
            else:
                regularite_range = None
    
    # Application des filtres
    df_explore = df.copy()
    
    if regions_filter and 'region' in df.columns:
        df_explore = df_explore[df_explore['region'].isin(regions_filter)]
    
    if annees_filter and 'annee' in df.columns:
        df_explore = df_explore[df_explore['annee'].isin(annees_filter)]
    
    if regularite_range and 'taux_regularite' in df.columns:
        df_explore = df_explore[
            (df_explore['taux_regularite'] >= regularite_range[0]) &
            (df_explore['taux_regularite'] <= regularite_range[1])
        ]
    
    # Affichage des résultats
    st.markdown(f"### 📊 Résultats : {len(df_explore):,} enregistrements")
    
    # Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Lignes", f"{len(df_explore):,}")
    
    with col2:
        if 'region' in df_explore.columns:
            st.metric("🗺️ Régions", df_explore['region'].nunique())
    
    with col3:
        if 'taux_regularite' in df_explore.columns:
            st.metric("📊 Régularité Moy.", f"{df_explore['taux_regularite'].mean():.1f}%")
    
    with col4:
        st.metric("💾 Taille", f"{df_explore.memory_usage(deep=True).sum() / 1024:.0f} KB")
    
    # Tableau de données
    st.markdown("### 📋 Tableau de Données")
    
    # Options d'affichage
    col1, col2 = st.columns(2)
    
    with col1:
        columns_to_show = st.multiselect(
            "Colonnes à afficher",
            options=df_explore.columns.tolist(),
            default=df_explore.columns.tolist()[:8]
        )
    
    with col2:
        rows_per_page = st.selectbox("Lignes par page", [10, 25, 50, 100], index=1)
    
    # Affichage paginé
    if columns_to_show:
        st.dataframe(
            df_explore[columns_to_show].head(rows_per_page),
            use_container_width=True,
            height=400
        )
    
    # Export
    st.markdown("### 💾 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_explore.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"ter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Note: Excel export nécessite un buffer
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_explore.to_excel(writer, index=False, sheet_name='TER_Data')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Télécharger Excel",
            data=buffer,
            file_name=f"ter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🌦️ ANALYSE MÉTÉO
# ═══════════════════════════════════════════════════════════════════════

elif page == "🌦️ Analyse Météo":
    st.title("🌦️ Analyse de l'Impact Météorologique")
    
    st.markdown("""
    Cette section analyse la corrélation entre les conditions météorologiques et les retards/annulations de trains.
    
    **Sources de données météo disponibles :**
    - 🌍 **Open-Meteo** : API gratuite, données historiques complètes
    - 🌤️ **OpenWeatherMap** : Nécessite une clé API (optionnel, plus précis)
    """)
    
    # Initialiser l'analyseur météo
    if 'weather_analyzer' not in st.session_state:
        st.session_state.weather_analyzer = WeatherAnalyzer(df)
    
    weather_analyzer = st.session_state.weather_analyzer
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1 : ENRICHISSEMENT DES DONNÉES
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("📥 Étape 1 : Enrichissement avec Données Météo")
    
    with st.expander("⚙️ Configuration de l'enrichissement", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.slider(
                "Nombre d'enregistrements à enrichir",
                min_value=100,
                max_value=min(5000, len(df)),
                value=min(1000, len(df)),
                step=100,
                help="⚠️ Plus le nombre est élevé, plus l'enrichissement prendra du temps (limite API)"
            )
        
        with col2:
            use_openweather = st.checkbox(
                "Utiliser OpenWeatherMap (optionnel)",
                help="Nécessite une clé API. Plus précis mais limité en appels gratuits."
            )
            
            openweather_key = None
            if use_openweather:
                openweather_key = st.text_input(
                    "Clé API OpenWeatherMap",
                    type="password",
                    help="Obtenez une clé gratuite sur https://openweathermap.org/api"
                )
        
        st.info("""
        💡 **Comment ça marche ?**
        
        1. L'application récupère les données météo historiques pour chaque date
        2. Les données sont associées aux régions via les grandes villes
        3. Un score de sévérité météo (0-100) est calculé
        4. Les corrélations avec les retards sont analysées
        
        ⏱️ **Temps estimé** : ~5-10 minutes pour 1000 enregistrements
        """)
        
        enrich_button = st.button(
            "🚀 Lancer l'enrichissement météo",
            type="primary",
            use_container_width=True
        )
    
    if enrich_button:
        with st.spinner("🌦️ Enrichissement en cours... Cela peut prendre plusieurs minutes."):
            df_enriched = weather_analyzer.enrich_with_weather(
                sample_size=sample_size,
                use_api_key=openweather_key if use_openweather else None
            )
            
            if df_enriched is not None:
                st.session_state.df_enriched = df_enriched
                st.success("✅ Enrichissement terminé avec succès !")
                st.balloons()
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2 : VISUALISATION DES DONNÉES MÉTÉO
    # ═══════════════════════════════════════════════════════════════════
    
    if 'df_enriched' in st.session_state:
        df_enriched = st.session_state.df_enriched
        
        st.markdown("---")
        st.subheader("📊 Étape 2 : Aperçu des Données Météo")
        
        # Statistiques météo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'temperature_mean' in df_enriched.columns:
                avg_temp = df_enriched['temperature_mean'].mean()
                st.metric(
                    "🌡️ Température Moyenne",
                    f"{avg_temp:.1f}°C"
                )
        
        with col2:
            if 'precipitation' in df_enriched.columns:
                total_precip = df_enriched['precipitation'].sum()
                st.metric(
                    "🌧️ Précipitations Totales",
                    f"{total_precip:.0f} mm"
                )
        
        with col3:
            if 'snow' in df_enriched.columns:
                jours_neige = (df_enriched['snow'] > 0).sum()
                st.metric(
                    "❄️ Jours avec Neige",
                    f"{jours_neige}"
                )
        
        with col4:
            if 'wind_gusts' in df_enriched.columns:
                max_wind = df_enriched['wind_gusts'].max()
                st.metric(
                    "💨 Rafale Max",
                    f"{max_wind:.0f} km/h"
                )
        
        # Aperçu du dataset enrichi
        st.markdown("### 📋 Aperçu des Données Enrichies")
        
        weather_cols = [col for col in df_enriched.columns if col in [
            'date', 'region', 'city', 'taux_regularite',
            'temperature_mean', 'precipitation', 'snow', 'wind_speed',
            'weather_severity_score'
        ]]
        
        if weather_cols:
            st.dataframe(
                df_enriched[weather_cols].head(20),
                use_container_width=True
            )
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 3 : ANALYSE DE CORRÉLATION
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("🔬 Étape 3 : Analyse des Corrélations")
        
        analyze_button = st.button(
            "📊 Analyser l'Impact Météo",
            type="primary",
            use_container_width=True
        )
        
        if analyze_button:
            with st.spinner("🔬 Analyse statistique en cours..."):
                results = weather_analyzer.analyze_weather_impact()
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.session_state.weather_results = results
        
        # Affichage des résultats
        if 'weather_results' in st.session_state:
            results = st.session_state.weather_results
            
            # Corrélation principale
            if 'correlation_regularite_meteo' in results:
                corr_data = results['correlation_regularite_meteo']
                
                st.markdown("### 📈 Corrélation Globale")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Affichage de la corrélation
                    corr_value = corr_data['correlation']
                    
                    # Couleur selon la force de la corrélation
                    if abs(corr_value) < 0.3:
                        color = "blue"
                    elif abs(corr_value) < 0.6:
                        color = "orange"
                    else:
                        color = "red"
                    
                    st.markdown(f"""
                    <div style='padding: 2rem; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {color};'>
                        <h3 style='margin: 0; color: {color};'>Coefficient de Corrélation : {corr_value:.3f}</h3>
                        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>{corr_data['interpretation']}</p>
                        <p style='margin: 0.5rem 0 0 0; color: #666;'>
                            Significativité : <strong>{corr_data['significance']}</strong> (p = {corr_data['p_value']:.4f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.info("""
                    **Interprétation :**
                    
                    - **< 0.3** : Faible corrélation
                    - **0.3-0.6** : Corrélation modérée
                    - **> 0.6** : Forte corrélation
                    
                    Une corrélation **négative** signifie que la régularité diminue quand la sévérité météo augmente.
                    """)
            
            # Impact de la neige
            if 'impact_neige' in results:
                st.markdown("### ❄️ Impact de la Neige")
                
                impact = results['impact_neige']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sans Neige",
                        f"{impact['regularite_sans_neige']:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Avec Neige",
                        f"{impact['regularite_avec_neige']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Perte de Régularité",
                        f"{impact['difference']:.2f}%",
                        delta=None
                    )
                
                if impact['difference'] > 5:
                    st.warning(f"⚠️ **Impact significatif** : La neige réduit la régularité de {impact['difference']:.1f} points de pourcentage.")
                elif impact['difference'] > 2:
                    st.info(f"📊 **Impact modéré** : La neige affecte la régularité de {impact['difference']:.1f} points.")
                else:
                    st.success(f"✅ **Impact faible** : La neige a un effet limité sur la régularité.")
            
            # Impact du vent
            if 'impact_vent' in results:
                st.markdown("### 💨 Impact du Vent Fort (> 90 km/h)")
                
                impact = results['impact_vent']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Vent Normal",
                        f"{impact['regularite_vent_normal']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Vent Fort",
                        f"{impact['regularite_vent_fort']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Perte de Régularité",
                        f"{impact['difference']:.2f}%"
                    )
            
            # Impact de la pluie
            if 'impact_pluie' in results:
                st.markdown("### 🌧️ Impact de la Pluie Forte (> 10 mm)")
                
                impact = results['impact_pluie']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Pluie Faible",
                        f"{impact['regularite_pluie_faible']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Pluie Forte",
                        f"{impact['regularite_pluie_forte']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Perte de Régularité",
                        f"{impact['difference']:.2f}%"
                    )
            
            # Retards par condition météo
            if 'retards_par_meteo' in results:
                st.markdown("### 📊 Retards Moyens par Condition Météo")
                
                retards = results['retards_par_meteo']
                
                df_plot = pd.DataFrame({
                    'Condition': list(retards.keys()),
                    'Retards Moyens': list(retards.values())
                })
                
                fig = px.bar(
                    df_plot,
                    x='Condition',
                    y='Retards Moyens',
                    color='Condition',
                    color_discrete_map={
                        'Bonne': 'green',
                        'Correcte': 'yellow',
                        'Difficile': 'orange',
                        'Extrême': 'red'
                    },
                    title="Nombre Moyen de Trains en Retard selon les Conditions Météo"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 4 : VISUALISATIONS AVANCÉES
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("📈 Étape 4 : Visualisations Avancées")
        
        viz_button = st.button(
            "🎨 Générer les Visualisations",
            type="primary",
            use_container_width=True
        )
        
        if viz_button:
            weather_analyzer.plot_weather_impact()
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 5 : EXPORT DES DONNÉES ENRICHIES
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("💾 Étape 5 : Export des Données Enrichies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            csv_enriched = df_enriched.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger CSV (avec météo)",
                data=csv_enriched,
                file_name=f"ter_enriched_weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export Excel
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_enriched.to_excel(writer, index=False, sheet_name='TER_Weather_Data')
                
                # Ajouter une feuille avec les résultats d'analyse
                if 'weather_results' in st.session_state:
                    results_df = pd.DataFrame([st.session_state.weather_results])
                    results_df.to_excel(writer, index=False, sheet_name='Analysis_Results')
            
            buffer.seek(0)
            
            st.download_button(
                label="📥 Télécharger Excel (avec météo)",
                data=buffer,
                file_name=f"ter_enriched_weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 6 : RAPPORT AUTOMATIQUE
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("📄 Rapport Automatique")
        
        if st.button("📝 Générer le Rapport d'Analyse", use_container_width=True):
            with st.spinner("📝 Génération du rapport..."):
                report = generate_weather_report(df_enriched, st.session_state.get('weather_results', {}))
                
                st.markdown(report)
                
                # Télécharger le rapport
                st.download_button(
                    label="💾 Télécharger le Rapport (Markdown)",
                    data=report,
                    file_name=f"rapport_meteo_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
    
    else:
        st.info("👆 Commencez par enrichir le dataset avec les données météo pour accéder aux analyses.")


    # ═══════════════════════════════════════════════════════════════════════
    # FONCTION UTILITAIRE : GÉNÉRATION DE RAPPORT
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_weather_report(df: pd.DataFrame, results: dict) -> str:
        """Génère un rapport markdown de l'analyse météo"""
        
        report = f"""
    # 🌦️ Rapport d'Analyse - Impact Météorologique sur la Ponctualité TER
    
    **Date du rapport** : {datetime.now().strftime('%d/%m/%Y %H:%M')}  
    **Période analysée** : {df['date'].min().strftime('%d/%m/%Y')} - {df['date'].max().strftime('%d/%m/%Y')}  
    **Nombre d'enregistrements** : {len(df):,}
    
    ---
    
    ## 📊 Résumé Exécutif
    
    """
        
        if 'correlation_regularite_meteo' in results:
            corr = results['correlation_regularite_meteo']
            report += f"""
    ### Corrélation Météo-Régularité
    
    - **Coefficient de corrélation** : {corr['correlation']:.3f}
    - **Interprétation** : {corr['interpretation']}
    - **Significativité statistique** : {corr['significance']} (p-value = {corr['p_value']:.4f})
    
    """
        
        if 'impact_neige' in results:
            neige = results['impact_neige']
            report += f"""
    ### ❄️ Impact de la Neige
    
    - Régularité sans neige : **{neige['regularite_sans_neige']:.2f}%**
    - Régularité avec neige : **{neige['regularite_avec_neige']:.2f}%**
    - **Perte de régularité : {neige['difference']:.2f} points**
    
    """
        
        if 'impact_vent' in results:
            vent = results['impact_vent']
            report += f"""
    ### 💨 Impact du Vent Fort
    
    - Régularité (vent normal) : **{vent['regularite_vent_normal']:.2f}%**
    - Régularité (vent fort >90km/h) : **{vent['regularite_vent_fort']:.2f}%**
    - **Perte de régularité : {vent['difference']:.2f} points**
    
    """
        
        if 'impact_pluie' in results:
            pluie = results['impact_pluie']
            report += f"""
    ### 🌧️ Impact de la Pluie Forte
    
    - Régularité (pluie faible) : **{pluie['regularite_pluie_faible']:.2f}%**
    - Régularité (pluie forte >10mm) : **{pluie['regularite_pluie_forte']:.2f}%**
    - **Perte de régularité : {pluie['difference']:.2f} points**
    
    """
        
        report += f"""
    ---
    
    ## 📈 Données Météorologiques
    
    ### Statistiques Générales
    
    """
        
        if 'temperature_mean' in df.columns:
            report += f"- **Température moyenne** : {df['temperature_mean'].mean():.1f}°C\n"
            report += f"- **Température minimale** : {df['temperature_mean'].min():.1f}°C\n"
            report += f"- **Température maximale** : {df['temperature_mean'].max():.1f}°C\n\n"
        
        if 'precipitation' in df.columns:
            report += f"- **Précipitations totales** : {df['precipitation'].sum():.0f} mm\n"
            report += f"- **Jours avec pluie** : {(df['precipitation'] > 0).sum()}\n\n"
        
        if 'snow' in df.columns:
            report += f"- **Jours avec neige** : {(df['snow'] > 0).sum()}\n\n"
        
        report += """
    ---
    
    ## 🎯 Recommandations
    
    ### Actions Prioritaires
    
    1. **Renforcer la prévention lors d'épisodes neigeux**
       - Anticiper les perturbations avec des prévisions météo en temps réel
       - Prépositionner les équipes de maintenance
    
    2. **Adapter la circulation en cas de vent fort**
       - Mettre en place des alertes automatiques
       - Réduire la vitesse des trains préventivement
    
    3. **Améliorer la communication voyageurs**
       - Informer en amont des perturbations météo prévues
       - Proposer des alternatives de transport
    
    ### Suivi et Monitoring
    
    - Continuer l'analyse mensuelle de l'impact météo
    - Développer un modèle prédictif de retards basé sur les prévisions météo
    - Créer un dashboard temps réel météo-ponctualité
    
    ---
    
    **Rapport généré automatiquement par l'Assistant IA TER SNCF**
    """
        
        return report

# ═══════════════════════════════════════════════════════════════════════
# PAGE : ⚙️ PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════

elif page == "⚙️ Paramètres":
    st.title("⚙️ Paramètres de l'Application")
    
    st.markdown("### 🔧 Configuration Actuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Azure Blob Storage**
        - Conteneur : `{Config.AZURE_CONTAINER_NAME}`
        - Fichier : `{Config.AZURE_BLOB_NAME}`
        - Connexion : {'✅ Configurée' if Config.AZURE_CONNECTION_STRING else '❌ Non configurée'}
        """)
    
    with col2:
        st.success(f"""
        **Mistral AI**
        - Modèle : `{Config.MISTRAL_MODEL}`
        - API Key : {'✅ Configurée' if Config.MISTRAL_API_KEY else '❌ Non configurée'}
        - Agent IA : {'✅ Actif' if st.session_state.agent else '❌ Inactif'}
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Informations sur le Dataset")
    
    st.dataframe(
        pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            '% Null': (df.isnull().sum() / len(df) * 100).round(2)
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.markdown("### 🔄 Actions de Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ Vider le cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache vidé !")
    
    with col3:
        if st.button("🔄 Réinitialiser l'app", use_container_width=True):
            st.session_state.clear()
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ À propos")
    
    st.info(f"""
    **Assistant IA - Analyse TER SNCF**
    
    Version : 1.0.0  
    Développé pour : SNCF Voyageurs  
    Technologie : Streamlit + LangChain + Mistral AI  
    Données : SNCF Open Data
    
    © 2025 - Projet de stage
    """)

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🚆 Assistant IA TER SNCF | Propulsé par Streamlit, LangChain & Mistral AI
    </div>
    """,
    unsafe_allow_html=True
)
