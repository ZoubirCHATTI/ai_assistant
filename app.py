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
#from data_loader import load_data_from_azure, get_data_summary
from data_loader import TERDataLoader  # ✅ Import correct
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
from data_loader import TERDataLoader  

# Au début de app.py, après les imports

# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TER Analysis Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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

# ═══════════════════════════════════════════════════════════════════════
# INITIALISATION DE LA SESSION
# ═══════════════════════════════════════════════════════════════════════

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'df_enriched' not in st.session_state:
    st.session_state.df_enriched = None

if 'current_df_hash' not in st.session_state:
    st.session_state.current_df_hash = None

# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════

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

# Charger les données
with st.spinner("⏳ Chargement des données TER depuis l'API SNCF..."):
    df = load_ter_data()

# Vérifier que les données sont chargées
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

# Afficher un résumé
st.success(f"✅ {len(df):,} enregistrements chargés depuis l'API SNCF")

# Afficher des infos sur les données
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

# ═══════════════════════════════════════════════════════════════════════
# INITIALISATION DES ANALYSEURS
# ═══════════════════════════════════════════════════════════════════════

# Initialiser l'analyseur météo
weather_analyzer = WeatherAnalyzer(df)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════

# Page Accueil
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
    
    # Afficher quelques stats rapides
    st.subheader("📈 Aperçu rapide des données")
    
    if 'taux_regularite' in df.columns and 'region' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 5 régions
            top5 = df.groupby('region')['taux_regularite'].mean().nlargest(5)
            st.write("**🏆 Top 5 régions les plus régulières**")
            for i, (region, taux) in enumerate(top5.items(), 1):
                st.write(f"{i}. {region}: {taux:.2f}%")
        
        with col2:
            # Bottom 5 régions
            bottom5 = df.groupby('region')['taux_regularite'].mean().nsmallest(5)
            st.write("**⚠️ Top 5 régions les moins régulières**")
            for i, (region, taux) in enumerate(bottom5.items(), 1):
                st.write(f"{i}. {region}: {taux:.2f}%")


"""
# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_ter_data():
    """Charge les données TER avec cache"""
    try:
        loader = TERDataLoader()
        df = loader.load_data()  # Cette méthode calcule maintenant automatiquement le taux_regularite
        
        # Afficher les infos dans la console
        print(f"✅ Données chargées : {len(df)} lignes")
        if 'taux_regularite' in df.columns:
            avg_reg = df['taux_regularite'].mean()
            print(f"📊 Taux de régularité moyen : {avg_reg:.2f}%")
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        return None

# Charger les données
with st.spinner("⏳ Chargement des données TER..."):
    df = load_ter_data()

if df is None or len(df) == 0:
    st.error("❌ Impossible de charger les données TER")
    st.stop()

st.success(f"✅ {len(df):,} enregistrements chargés")

# Afficher les infos sur le taux de régularité
if 'taux_regularite' in df.columns:
    avg_reg = df['taux_regularite'].mean()
    st.info(f"📊 Taux de régularité moyen calculé : **{avg_reg:.2f}%**")
else:
    st.warning("⚠️ Le taux de régularité n'a pas pu être calculé")

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
            "🌦️ Analyse Météo",  # ← NOUVELLE PAGE
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
"""
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
# PAGE : 💬 CHAT IA
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# PAGE : 💬 CHAT IA
# ═══════════════════════════════════════════════════════════════════════

elif page == "💬 Chat IA":
    st.title("💬 Chat avec l'Assistant IA")
    
    # ═══════════════════════════════════════════════════════════════════
    # SÉLECTION DU DATASET (avec ou sans météo)
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    
    # Vérifier si des données enrichies existent
    has_weather_data = 'df_enriched' in st.session_state and st.session_state.df_enriched is not None
    
    if has_weather_data:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.success("✅ Données météo détectées ! Le Chat IA peut analyser l'impact météorologique.")
        
        with col2:
            use_weather = st.toggle(
                "Utiliser données météo",
                value=True,
                help="Active/Désactive l'utilisation des données météo enrichies"
            )
        
        # Sélectionner le bon DataFrame
        if use_weather:
            df_for_agent = st.session_state.df_enriched
            st.info("🌦️ Mode : **Analyse avec données météo**")
        else:
            df_for_agent = df
            st.info("📊 Mode : **Analyse standard (sans météo)**")
    else:
        df_for_agent = df
        st.info("""
        📊 **Mode standard** - Données météo non disponibles
        
        Pour activer l'analyse météo :
        1. Allez dans **🌦️ Analyse Météo**
        2. Lancez l'enrichissement météo
        3. Revenez ici pour poser des questions sur la météo
        """)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════
    # INITIALISATION DE L'AGENT AVEC LE BON DATASET
    # ═══════════════════════════════════════════════════════════════════
    
    # Réinitialiser l'agent si le dataset a changé
    if 'current_df_id' not in st.session_state:
        st.session_state.current_df_id = None
    
    current_df_id = id(df_for_agent)
    
    if st.session_state.current_df_id != current_df_id or st.session_state.agent is None:
        with st.spinner("🔄 Initialisation de l'agent IA avec le nouveau dataset..."):
            try:
                st.session_state.agent = TERAnalysisAgent(df_for_agent)
                st.session_state.current_df_id = current_df_id
                st.success("✅ Agent IA initialisé")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'initialisation de l'agent : {e}")
                st.session_state.agent = None
    
    # ═══════════════════════════════════════════════════════════════════
    # INTERFACE DE CHAT
    # ═══════════════════════════════════════════════════════════════════
    
    if st.session_state.agent is None:
        st.error("❌ L'agent IA n'est pas disponible. Vérifiez votre clé API Mistral.")
    else:
        # Afficher les capacités de l'agent
        with st.expander("💡 Ce que je peux faire", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Analyses Standard :**
                - Calculer la régularité globale
                - Comparer les régions
                - Analyser l'évolution temporelle
                - Statistiques sur les trains
                - Top/Flop des régions
                """)
            
            with col2:
                if has_weather_data and use_weather:
                    st.markdown("""
                    **🌦️ Analyses Météo :**
                    - Impact de la neige
                    - Impact du vent fort
                    - Jours avec météo extrême
                    - Corrélation météo-régularité
                    - Conditions météo par région
                    """)
                else:
                    st.markdown("""
                    **🌦️ Analyses Météo :**
                    
                    ⚠️ Non disponible
                    
                    Enrichissez d'abord le dataset
                    dans la page Analyse Météo
                    """)
        
        # Exemples de questions
        with st.expander("💬 Exemples de questions", expanded=False):
            st.markdown("""
            **Questions Standard :**
            - Quelle est la régularité moyenne globale ?
            - Quelle région a la meilleure ponctualité ?
            - Combien de trains ont été supprimés ?
            - Quelles sont les 5 pires régions ?
            - Compare 2023 et 2024
            
            **Questions Météo** (si données disponibles) :
            - Est-ce que la neige affecte les trains ?
            - Quel est l'impact du vent fort ?
            - Quels sont les jours avec météo extrême ?
            - Y a-t-il une corrélation entre météo et retards ?
            - Compare l'impact de la neige vs le vent
            """)
        
        st.markdown("---")
        
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
                    st.exception(e)
        
        # Boutons d'action
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Effacer l'historique", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Réinitialiser l'agent", use_container_width=True):
                st.session_state.agent = None
                st.session_state.current_df_id = None
                st.rerun()
        
        with col3:
            if has_weather_data:
                if st.button("🌦️ Basculer mode météo", use_container_width=True):
                    st.rerun()
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
    
    # Vérifier si le module météo existe
    try:
        from weather_analyzer import WeatherAnalyzer
        module_meteo_disponible = True
    except ImportError:
        module_meteo_disponible = False
        st.error("""
        ❌ **Module météo non trouvé !**
        
        Le fichier `weather_analyzer.py` n'a pas été trouvé. 
        
        **Actions requises :**
        1. Créez le fichier `weather_analyzer.py` dans le même dossier que `app.py`
        2. Copiez-y le code du module météo fourni précédemment
        3. Rechargez l'application
        """)
        st.stop()
    
    # Initialiser l'analyseur météo
    if 'weather_analyzer' not in st.session_state:
        try:
            st.session_state.weather_analyzer = WeatherAnalyzer(df)
            st.success("✅ Analyseur météo initialisé")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation : {e}")
            st.session_state.weather_analyzer = None
    
    weather_analyzer = st.session_state.weather_analyzer
    
    if weather_analyzer is None:
        st.warning("⚠️ L'analyseur météo n'est pas disponible.")
        st.stop()
    
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
                value=min(500, len(df)),
                step=100,
                help="⚠️ Plus le nombre est élevé, plus l'enrichissement prendra du temps (appels API)"
            )
            
            st.info(f"""
            **Estimation :**
            - Enregistrements : {sample_size}
            - Temps estimé : ~{sample_size // 100} minutes
            - Appels API : ~{sample_size // 10}
            """)
        
        with col2:
            use_openweather = st.checkbox(
                "🌤️ Utiliser OpenWeatherMap (optionnel)",
                help="Nécessite une clé API. Plus précis mais limité en appels gratuits.",
                value=False
            )
            
            openweather_key = None
            if use_openweather:
                openweather_key = st.text_input(
                    "Clé API OpenWeatherMap",
                    type="password",
                    help="Obtenez une clé gratuite sur https://openweathermap.org/api"
                )
                
                if not openweather_key:
                    st.warning("⚠️ Entrez une clé API pour utiliser OpenWeatherMap")
        
        st.markdown("---")
        
        st.markdown("""
        ### 💡 Comment ça marche ?
        
        1. **Récupération météo** : L'application récupère les données historiques pour chaque date
        2. **Géolocalisation** : Les données sont associées aux régions via les grandes villes
        3. **Score de sévérité** : Un score météo (0-100) est calculé automatiquement
        4. **Analyse statistique** : Les corrélations avec les retards sont calculées
        
        ### 📊 Données météo collectées :
        - 🌡️ Températures (min, max, moyenne)
        - 🌧️ Précipitations et pluie
        - ❄️ Chutes de neige
        - 💨 Vitesse du vent et rafales
        - 📈 Score de sévérité global
        """)
        
        enrich_button = st.button(
            "🚀 Lancer l'enrichissement météo",
            type="primary",
            use_container_width=True,
            help="Lance la récupération des données météo (peut prendre plusieurs minutes)"
        )
    
    if enrich_button:
        if 'date' not in df.columns:
            st.error("❌ La colonne 'date' est nécessaire pour l'enrichissement météo.")
        else:
            progress_container = st.container()
            
            with progress_container:
                st.info("🌦️ **Enrichissement en cours...** Cela peut prendre plusieurs minutes selon le nombre d'enregistrements.")
                
                try:
                    df_enriched = weather_analyzer.enrich_with_weather(
                        sample_size=sample_size,
                        use_api_key=openweather_key if use_openweather else None
                    )
                    
                    if df_enriched is not None and len(df_enriched) > 0:
                        st.session_state.df_enriched = df_enriched
                        st.session_state.weather_analyzer.df_enriched = df_enriched
                        
                        st.success("✅ **Enrichissement terminé avec succès !**")
                        st.balloons()
                        
                        # Afficher un aperçu
                        st.markdown("### 📋 Aperçu des premières lignes enrichies")
                        weather_cols = [col for col in df_enriched.columns if col in [
                            'date', 'region', 'temperature_mean', 'precipitation', 
                            'snow', 'wind_speed', 'weather_severity_score'
                        ]]
                        st.dataframe(df_enriched[weather_cols].head(10), use_container_width=True)
                    else:
                        st.error("❌ L'enrichissement a échoué ou n'a retourné aucune donnée.")
                        
                except Exception as e:
                    st.error(f"❌ **Erreur lors de l'enrichissement :** {e}")
                    st.exception(e)
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2 : VISUALISATION DES DONNÉES MÉTÉO
    # ═══════════════════════════════════════════════════════════════════
    
    if 'df_enriched' in st.session_state and st.session_state.df_enriched is not None:
        df_enriched = st.session_state.df_enriched
        
        st.markdown("---")
        st.subheader("📊 Étape 2 : Aperçu des Données Météo")
        
        # KPIs météo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'temperature_mean' in df_enriched.columns:
                avg_temp = df_enriched['temperature_mean'].dropna().mean()
                st.metric(
                    "🌡️ Température Moyenne",
                    f"{avg_temp:.1f}°C"
                )
        
        with col2:
            if 'precipitation' in df_enriched.columns:
                total_precip = df_enriched['precipitation'].dropna().sum()
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
                max_wind = df_enriched['wind_gusts'].dropna().max()
                st.metric(
                    "💨 Rafale Maximale",
                    f"{max_wind:.0f} km/h" if not pd.isna(max_wind) else "N/A"
                )
        
        # Tableau détaillé
        st.markdown("### 📋 Tableau des Données Enrichies")
        
        weather_cols = [col for col in df_enriched.columns if col in [
            'date', 'region', 'city', 'taux_regularite',
            'temperature_mean', 'precipitation', 'rain', 'snow', 
            'wind_speed', 'wind_gusts', 'weather_severity_score'
        ]]
        
        if weather_cols:
            st.dataframe(
                df_enriched[weather_cols].head(50),
                use_container_width=True,
                height=400
            )
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 3 : ANALYSE DE CORRÉLATION
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("🔬 Étape 3 : Analyse des Corrélations Météo-Retards")
        
        st.markdown("""
        Cette analyse utilise des tests statistiques pour déterminer si les conditions météorologiques 
        ont un impact significatif sur la ponctualité des trains.
        """)
        
        analyze_button = st.button(
            "📊 Lancer l'Analyse Statistique",
            type="primary",
            use_container_width=True
        )
        
        if analyze_button:
            with st.spinner("🔬 Analyse statistique en cours..."):
                try:
                    results = weather_analyzer.analyze_weather_impact()
                    
                    if 'error' in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        st.session_state.weather_results = results
                        st.success("✅ Analyse terminée !")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")
                    st.exception(e)
        
        # Affichage des résultats d'analyse
        if 'weather_results' in st.session_state:
            results = st.session_state.weather_results
            
            st.markdown("---")
            st.markdown("## 📈 Résultats de l'Analyse")
            
            # Corrélation principale
            if 'correlation_regularite_meteo' in results:
                corr_data = results['correlation_regularite_meteo']
                
                st.markdown("### 📊 Corrélation Globale Météo-Régularité")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    corr_value = corr_data['correlation']
                    
                    # Couleur selon la force
                    if abs(corr_value) < 0.3:
                        color = "#2196F3"  # Bleu
                        niveau = "Faible"
                    elif abs(corr_value) < 0.6:
                        color = "#FF9800"  # Orange
                        niveau = "Modérée"
                    else:
                        color = "#F44336"  # Rouge
                        niveau = "Forte"
                    
                    st.markdown(f"""
                    <div style='padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                                border-radius: 15px; border-left: 8px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='margin: 0; color: {color}; font-size: 2.5rem;'>r = {corr_value:.3f}</h2>
                        <p style='margin: 1rem 0 0 0; font-size: 1.3rem; font-weight: bold;'>{corr_data['interpretation']}</p>
                        <p style='margin: 0.5rem 0 0 0; color: #555; font-size: 1.1rem;'>
                            Significativité : <strong>{corr_data['significance']}</strong> (p = {corr_data['p_value']:.4f})
                        </p>
                        <p style='margin: 0.5rem 0 0 0; color: #666;'>
                            Niveau de corrélation : <strong>{niveau}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.info("""
                    **📚 Interprétation du coefficient :**
                    
                    - **|r| < 0.3** : Corrélation faible
                    - **0.3 ≤ |r| < 0.6** : Corrélation modérée
                    - **|r| ≥ 0.6** : Corrélation forte
                    
                    Une corrélation **négative** signifie que la régularité diminue quand la sévérité météo augmente.
                    
                    **p-value < 0.05** = statistiquement significatif
                    """)
            
            # Impact de la neige
            if 'impact_neige' in results:
                st.markdown("---")
                st.markdown("### ❄️ Impact de la Neige sur la Régularité")
                
                impact = results['impact_neige']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sans Neige",
                        f"{impact['regularite_sans_neige']:.2f}%",
                        help="Taux de régularité moyen les jours sans neige"
                    )
                
                with col2:
                    st.metric(
                        "Avec Neige ❄️",
                        f"{impact['regularite_avec_neige']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse",
                        help="Taux de régularité moyen les jours avec neige"
                    )
                
                with col3:
                    st.metric(
                        "Perte de Régularité",
                        f"{impact['difference']:.2f} points",
                        help="Différence de régularité due à la neige"
                    )
                
                # Interprétation
                if impact['difference'] > 5:
                    st.error(f"⚠️ **Impact significatif** : La neige réduit la régularité de **{impact['difference']:.1f} points** de pourcentage.")
                elif impact['difference'] > 2:
                    st.warning(f"📊 **Impact modéré** : La neige affecte la régularité de **{impact['difference']:.1f} points**.")
                else:
                    st.success(f"✅ **Impact faible** : La neige a un effet limité (**{impact['difference']:.1f} points**).")
            
            # Impact du vent
            if 'impact_vent' in results:
                st.markdown("---")
                st.markdown("### 💨 Impact du Vent Fort (> 90 km/h)")
                
                impact = results['impact_vent']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vent Normal", f"{impact['regularite_vent_normal']:.2f}%")
                
                with col2:
                    st.metric(
                        "Vent Fort 💨",
                        f"{impact['regularite_vent_fort']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric("Perte", f"{impact['difference']:.2f} points")
            
            # Impact de la pluie
            if 'impact_pluie' in results:
                st.markdown("---")
                st.markdown("### 🌧️ Impact de la Pluie Forte (> 10 mm)")
                
                impact = results['impact_pluie']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pluie Faible", f"{impact['regularite_pluie_faible']:.2f}%")
                
                with col2:
                    st.metric(
                        "Pluie Forte 🌧️",
                        f"{impact['regularite_pluie_forte']:.2f}%",
                        delta=f"-{impact['difference']:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric("Perte", f"{impact['difference']:.2f} points")
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 4 : VISUALISATIONS
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("📈 Étape 4 : Visualisations Avancées")
        
        if st.button("🎨 Générer les Graphiques", type="primary", use_container_width=True):
            try:
                weather_analyzer.plot_weather_impact()
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération des visualisations : {e}")
                st.exception(e)
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 5 : EXPORT
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("💾 Étape 5 : Export des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_enriched.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger CSV (avec météo)",
                data=csv,
                file_name=f"ter_meteo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_enriched.to_excel(writer, index=False, sheet_name='Donnees_Meteo')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Télécharger Excel (avec météo)",
                data=buffer,
                file_name=f"ter_meteo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    else:
        st.info("👆 **Commencez par enrichir le dataset** avec les données météo pour accéder aux analyses et visualisations.")
        
        st.markdown("""
        ### 🎯 Prochaines étapes :
        
        1. **Configurez** le nombre d'enregistrements à analyser
        2. **Lancez** l'enrichissement météo (bouton ci-dessus)
        3. **Attendez** quelques minutes pendant la récupération des données
        4. **Analysez** les corrélations et visualisations
        5. **Exportez** les résultats enrichis
        """)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🚆 Assistant IA TER SNCF | Propulsé par Streamlit, LangChain & Mistral AI
    </div>
    """,
    unsafe_allow_html=True
)
# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

