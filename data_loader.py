# data_loader.py
# -*- coding: utf-8 -*-
"""
Module de chargement des données TER.
Charge depuis Azure Blob Storage en priorité, avec fallback sur l'API SNCF.
"""

import io
import requests
import pandas as pd
from typing import Optional

from config import Config


class TERDataLoader:
    """Charge et prépare les données TER depuis Azure ou l'API SNCF"""

    SNCF_API_URL = (
        "https://ressources.data.sncf.com/api/explore/v2.1"
        "/catalog/datasets/regularite-mensuelle-ter/records"
    )

    def load_data(self, max_records: int = None) -> pd.DataFrame:
        """
        Charge les données TER.
        Priorité : Azure Blob Storage → API SNCF publique.

        Args:
            max_records: Limite le nombre d'enregistrements (None = tous).

        Returns:
            DataFrame nettoyé avec taux_regularite calculé.
        """
        df = None

        # 1. Tentative Azure Blob Storage
        if Config.AZURE_CONNECTION_STRING:
            print("☁️ Tentative de chargement depuis Azure Blob Storage...")
            df = self._load_from_azure()

        # 2. Fallback API SNCF
        if df is None or len(df) == 0:
            print("🌐 Chargement depuis l'API SNCF publique...")
            df = self._load_from_sncf_api(max_records)

        if df is None or len(df) == 0:
            raise ValueError("❌ Impossible de charger les données TER (Azure et API SNCF ont échoué)")

        df = self._clean_data(df)
        df = self._calculate_regularite(df)

        print(f"✅ {len(df):,} enregistrements disponibles")
        return df

    def _load_from_azure(self) -> Optional[pd.DataFrame]:
        """Charge le fichier Excel depuis Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient

            client = BlobServiceClient.from_connection_string(Config.AZURE_CONNECTION_STRING)
            blob = client.get_blob_client(
                container=Config.AZURE_CONTAINER_NAME,
                blob=Config.AZURE_BLOB_NAME
            )

            data = blob.download_blob().readall()
            blob_name = Config.AZURE_BLOB_NAME.lower()

            if blob_name.endswith(".xlsx") or blob_name.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(data))
            elif blob_name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data))
            else:
                # Essayer Excel par défaut
                df = pd.read_excel(io.BytesIO(data))

            print(f"   ✅ Azure : {len(df):,} lignes × {len(df.columns)} colonnes chargées")
            return df

        except ImportError:
            print("   ⚠️ azure-storage-blob non installé, passage à l'API SNCF")
            return None
        except Exception as e:
            print(f"   ⚠️ Erreur Azure ({type(e).__name__}) : {e}")
            return None

    def _load_from_sncf_api(self, max_records: int = None) -> Optional[pd.DataFrame]:
        """Charge les données depuis l'API SNCF Open Data"""
        params = {'limit': 100, 'offset': 0}
        all_records = []

        print("📥 Téléchargement depuis l'API SNCF...")

        while True:
            try:
                response = requests.get(self.SNCF_API_URL, params=params, timeout=30)
                response.raise_for_status()

                records = response.json().get('results', [])
                if not records:
                    break

                all_records.extend(records)
                print(f"   → {len(all_records):,} enregistrements téléchargés...")

                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    break

                if len(records) < params['limit']:
                    break

                params['offset'] += params['limit']

            except requests.exceptions.Timeout:
                print("   ⚠️ Timeout, nouvelle tentative...")
                continue
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Erreur API SNCF : {e}")
                break

        if not all_records:
            return None

        print(f"   ✅ API SNCF : {len(all_records):,} enregistrements téléchargés")
        return pd.DataFrame(all_records)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et type les colonnes du DataFrame"""
        print("\n🧹 Nettoyage des données...")
        df = df.copy()

        # Normaliser les noms de colonnes (minuscules, sans espaces)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Conversion de la date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            print(f"   ✅ Dates converties")

        # Colonnes numériques
        numeric_cols = [
            'nombre_trains_prevus', 'nombre_trains_circules',
            'nombre_trains_supprimes', 'nombre_trains_annules',
            'nombre_trains_retard', 'nombre_trains_a_l_heure',
            'taux_regularite'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Colonne region : nettoyage
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.strip()

        print(f"   ✅ Nettoyage terminé : {len(df):,} lignes conservées")
        return df

    def _calculate_regularite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule ou valide le taux de régularité"""
        print("\n📊 Calcul du taux de régularité...")
        df = df.copy()

        # Déjà présente et complète
        if 'taux_regularite' in df.columns:
            valides = df['taux_regularite'].notna().sum()
            if valides > len(df) * 0.8:
                avg = df['taux_regularite'].mean()
                print(f"   ✅ Colonne existante utilisée (moyenne : {avg:.2f}%)")
                df['taux_regularite'] = df['taux_regularite'].clip(0, 100)
                return df

        # Méthode 1 : trains à l'heure / trains circulés
        if 'nombre_trains_a_l_heure' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                df['nombre_trains_a_l_heure'] / df['nombre_trains_circules'] * 100
            ).clip(0, 100)
            print("   ✅ Calculé depuis nombre_trains_a_l_heure / nombre_trains_circules")

        # Méthode 2 : 100 - (retards / circulés)
        elif 'nombre_trains_retard' in df.columns and 'nombre_trains_circules' in df.columns:
            df['taux_regularite'] = (
                100 - df['nombre_trains_retard'] / df['nombre_trains_circules'] * 100
            ).clip(0, 100)
            print("   ✅ Calculé depuis nombre_trains_retard / nombre_trains_circules")

        # Méthode 3 : chercher une colonne similaire
        else:
            candidates = [c for c in df.columns if 'regularite' in c or 'ponctualite' in c]
            if candidates:
                df['taux_regularite'] = pd.to_numeric(df[candidates[0]], errors='coerce').clip(0, 100)
                print(f"   ✅ Colonne '{candidates[0]}' utilisée")
            else:
                print("   ⚠️ Impossible de calculer taux_regularite — colonne mise à 0")
                df['taux_regularite'] = 0.0

        avg = df['taux_regularite'].mean()
        print(f"   📈 Régularité moyenne : {avg:.2f}%")
        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Retourne un résumé du DataFrame"""
        info = {
            'nb_lignes': len(df),
            'nb_colonnes': len(df.columns),
            'colonnes': df.columns.tolist(),
            'has_regularite': 'taux_regularite' in df.columns,
            'has_region': 'region' in df.columns,
            'has_date': 'date' in df.columns,
        }

        if 'date' in df.columns:
            info['date_min'] = df['date'].min()
            info['date_max'] = df['date'].max()

        if 'region' in df.columns:
            info['nb_regions'] = df['region'].nunique()
            info['regions'] = sorted(df['region'].unique().tolist())

        if 'taux_regularite' in df.columns:
            info['regularite_moyenne'] = df['taux_regularite'].mean()
            info['regularite_min'] = df['taux_regularite'].min()
            info['regularite_max'] = df['taux_regularite'].max()

        return info


if __name__ == "__main__":
    loader = TERDataLoader()
    df = loader.load_data(max_records=500)
    info = loader.get_data_info(df)
    print(f"\n✅ {info['nb_lignes']:,} lignes | {info['nb_colonnes']} colonnes")
    if info['has_regularite']:
        print(f"📊 Régularité moyenne : {info['regularite_moyenne']:.2f}%")
    if info['has_region']:
        print(f"🗺️ {info['nb_regions']} régions")
