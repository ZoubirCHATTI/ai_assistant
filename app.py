# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
🚆 ASSISTANT IA - ANALYSE TER SNCF
Application Streamlit pour l'analyse intelligente des données TER
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import sys
from datetime import datetime

# Import des modules locaux
from config import Config, check_config
from data_loader import load_data_from_azure, get_data_summary
from visualizations import (
    plot_kpi_cards,
    plot_regularite_evolution,
    plot_regularite_by_region,
    plot_causes_retards,
    plot_heatmap_regularite,
    plot_custom_visualization
)
from ai_agent import TERAnalysisAgent
# Ajoute cet import avec les autres imports
from weather_analyzer import WeatherAnalyzer
# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #689f38;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# VÉRIFICATION DE LA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

check_config()

# ═══════════════════════════════════════════════════════════════════════
# INITIALISATION DE LA SESSION
# ═══════════════════════════════════════════════════════════════════════

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.agent = None
    st.session_state.chat_history = []

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/76/Logo_SNCF.svg", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Choisissez une page :",
        [
            "🏠 Accueil",
            "📊 Dashboard",
            "💬 Chat IA",
            "📈 Visualisations Personnalisées",
            "🔍 Explorateur de Données",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    
    # Bouton de rechargement des données
    if st.button("🔄 Recharger les données", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()
    
    st.markdown("---")
    st.caption(f"🕒 Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("🔒 Données sécurisées via Azure Blob Storage")

# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════

if not st.session_state.data_loaded:
    with st.spinner("🚂 Chargement des données TER depuis Azure..."):
        df = load_data_from_azure()
        
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Initialiser l'agent IA
            try:
                st.session_state.agent = TERAnalysisAgent(df)
            except Exception as e:
                st.warning(f"⚠️ Agent IA non disponible : {e}")
                st.session_state.agent = None
            
            st.success("✅ Données chargées avec succès !")
        else:
            st.error("❌ Impossible de charger les données. Vérifiez votre configuration Azure.")
            st.stop()

df = st.session_state.df

# ═══════════════════════════════════════════════════════════════════════
# PAGE : 🏠 ACCUEIL
# ═══════════════════════════════════════════════════════════════════════

if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🚆 Assistant IA - Analyse TER SNCF</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Votre compagnon intelligent pour l\'analyse des données ferroviaires</div>', unsafe_allow_html=True)
    
    # Résumé des données
    summary = get_data_summary(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        ### 📊 Dataset
        - **{summary['total_rows']:,}** enregistrements
        - **{summary['total_columns']}** colonnes
        - **{summary['memory_usage']:.2f}** MB en mémoire
        """)
    
    with col2:
        if summary['date_range']:
            min_date, max_date = summary['date_range']
            st.success(f"""
            ### 📅 Période
            - **Du** {min_date.strftime('%d/%m/%Y')}
            - **Au** {max_date.strftime('%d/%m/%Y')}
            - **{(max_date - min_date).days}** jours
            """)
        else:
            st.success("### 📅 Période\nDonnées disponibles")
    
    with col3:
        if summary['regions']:
            st.warning(f"""
            ### 🗺️ Couverture
            - **{len(summary['regions'])}** régions
            - Analyse nationale
            - Données temps réel
            """)
        else:
            st.warning("### 🗺️ Couverture\nAnalyse disponible")
    
    st.markdown("---")
    
    # Fonctionnalités
    st.subheader("✨ Fonctionnalités de l'Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Dashboard Interactif
        - KPIs en temps réel
        - Graphiques dynamiques
        - Analyse par région et période
        - Export des visualisations
        
        #### 💬 Chat IA Conversationnel
        - Questions en langage naturel
        - Réponses contextualisées
        - Historique des conversations
        - Suggestions intelligentes
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Visualisations Personnalisées
        - Créateur de graphiques
        - Multiples types de charts
        - Filtres avancés
        - Exports haute résolution
        
        #### 🔍 Explorateur de Données
        - Filtrage multi-critères
        - Recherche avancée
        - Export CSV/Excel
        - Statistiques détaillées
        """)
    
    st.markdown("---")
    
    # Guide de démarrage rapide
    st.subheader("🚀 Démarrage Rapide")
    
    with st.expander("📘 Comment utiliser l'Assistant ?", expanded=True):
        st.markdown("""
        ### 1️⃣ Explorez le Dashboard
        Accédez à la page **📊 Dashboard** pour voir une vue d'ensemble des KPIs et métriques clés.
        
        ### 2️⃣ Posez vos Questions
        Allez dans **💬 Chat IA** et posez vos questions en français :
        - "Quelle est la régularité moyenne ?"
        - "Quelles sont les pires régions ?"
        - "Combien de trains ont été supprimés ?"
        
        ### 3️⃣ Créez des Visualisations
        Dans **📈 Visualisations Personnalisées**, créez vos propres graphiques :
        - Choisissez le type de graphique
        - Sélectionnez les colonnes
        - Appliquez des filtres
        
        ### 4️⃣ Explorez les Données
        Utilisez **🔍 Explorateur de Données** pour :
        - Filtrer par région, date, etc.
        - Rechercher des valeurs spécifiques
        - Exporter vos sélections
        """)

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
    
    if st.session_state.agent is None:
        st.error("❌ L'agent IA n'est pas disponible. Vérifiez votre clé API Mistral.")
    else:
        st.markdown("""
        Posez vos questions en français sur les données TER. L'IA analysera les données et vous donnera des réponses précises.
        """)
        
        # Exemples de questions
        with st.expander("💡 Exemples de questions", expanded=False):
            st.markdown("""
            - Quelle est la régularité moyenne globale ?
            - Quelle région a la meilleure ponctualité ?
            - Combien de trains ont été supprimés ?
            - Montre-moi l'évolution de la régularité
            - Quelles sont les 5 pires régions ?
            - Donne-moi les statistiques sur les trains
            """)
        
        # Affichage de l'historique
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 **Vous** : {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant** : {message["content"]}</div>', unsafe_allow_html=True)
        
        # Zone de saisie
        user_question = st.chat_input("Posez votre question ici...")
        
        if user_question:
            # Ajouter la question à l'historique
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Obtenir la réponse de l'agent
            with st.spinner("🤔 L'IA réfléchit..."):
                try:
                    response = st.session_state.agent.ask(user_question)
                    
                    # Ajouter la réponse à l'historique
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()

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
# Ajoute cet import avec les autres imports
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/76/Logo_SNCF.svg", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Choisissez une page :",
        [
            "🏠 Accueil",
            "📊 Dashboard",
            "💬 Chat IA",
            "🌦️ Analyse Météo",  # ← NOUVELLE PAGE
            "📈 Visualisations Personnalisées",
            "🔍 Explorateur de Données",
            "⚙️ Paramètres"
        ]
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
