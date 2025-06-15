#!/usr/bin/env python3
"""
Logbuch Analyzer - Hauptmodul
CAS Data Science - Text Analysis Projekt

Intelligente Analyse von industriellen Logbuch-Einträgen mittels 
semantischer Embeddings und LLM-Integration.

"""

import pandas as pd
import numpy as np
import faiss
import requests
import json
import re
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict
import os
import chardet
import warnings
from io import StringIO

# Warnings unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"SentenceTransformers Import fehlgeschlagen: {e}")
    print("Installation mit: pip install sentence-transformers")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogbookAnalyzer:
    """
    Hauptklasse für intelligente Logbuch-Analyse.
    
    Funktionalitäten:
    - Robuste CSV-Datenladung mit automatischer Formaterkennung
    - Semantische Embedding-Erstellung mit E5-Modellen
    - Schnelle Ähnlichkeitssuche mit FAISS
    - KI-gestützte Analyse mit Ollama LLM
    """
    
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small'):
        """
        Initialisiert den Analyzer.
        
        Args:
            model_name: SentenceTransformer Modellname
        """
        self.model = None
        self.index = None
        self.df = None
        self.embeddings = None
        self.model_name = model_name
        self.ollama_model = "llama3.1:8b"
        
        logger.info(f"LogbookAnalyzer initialisiert mit Modell: {model_name}")
    
    def load_embedding_model(self) -> bool:
        """Lädt das Embedding-Modell mit Fallback-Strategien."""
        try:
            logger.info(f"Lade Embedding-Modell: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name, device='cpu')
            self.model.max_seq_length = 256
            
            logger.info(f"Erfolgreich geladen: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden von {self.model_name}: {e}")
            
            # Fallback-Modelle
            fallback_models = [
                'intfloat/multilingual-e5-base',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'sentence-transformers/all-MiniLM-L6-v2'
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Versuche Fallback: {fallback_model}")
                    self.model = SentenceTransformer(fallback_model, device='cpu')
                    self.model_name = fallback_model
                    logger.info(f"Fallback erfolgreich: {fallback_model}")
                    return True
                except Exception:
                    continue
            
            logger.error("Alle Embedding-Modelle fehlgeschlagen")
            return False
    
    def detect_encoding_and_separator(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Robuste Erkennung von Encoding und CSV-Trennzeichen."""
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            
            # Encoding-Erkennung
            detected = chardet.detect(raw_bytes)
            primary_encoding = detected.get('encoding', 'utf-8') if detected else 'utf-8'
            
            # Fallback-Encodings
            encodings_to_try = [primary_encoding, 'utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            encodings_to_try = list(dict.fromkeys(encodings_to_try))
            
            separators = [',', ';', '\t', '|']
            
            best_result = None
            max_columns = 0
            
            for encoding in encodings_to_try:
                try:
                    text_content = raw_bytes.decode(encoding)
                    
                    for separator in separators:
                        try:
                            test_io = StringIO(text_content)
                            test_df = pd.read_csv(test_io, sep=separator, nrows=50, on_bad_lines='skip')
                            
                            num_cols = len(test_df.columns)
                            if num_cols >= 3 and num_cols > max_columns:
                                max_columns = num_cols
                                best_result = (encoding, separator)
                        except:
                            continue
                except UnicodeDecodeError:
                    continue
            
            if best_result:
                encoding, separator = best_result
                logger.info(f"Erkannt: {encoding} / Trennzeichen: '{separator}'")
                return encoding, separator
            
            return None, None
            
        except Exception as e:
            logger.error(f"Fehler bei Encoding/Separator-Erkennung: {e}")
            return None, None

    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Lädt CSV-Daten mit robuster Formaterkennung.
        
        Args:
            file_path: Pfad zur CSV-Datei
            
        Returns:
            DataFrame oder None bei Fehler
        """
        try:
            encoding, separator = self.detect_encoding_and_separator(file_path)
            
            if not encoding or not separator:
                logger.error("Automatische Format-Erkennung fehlgeschlagen")
                return None
            
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding,
                on_bad_lines='skip',
                skipinitialspace=True,
                dtype=str
            )
            
            if len(df) == 0:
                logger.error("Keine Daten in CSV-Datei gefunden")
                return None
            
            logger.info(f"CSV geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim CSV-Laden: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Verarbeitet und bereinigt die Logbuch-Daten.
        
        Args:
            df: Rohe CSV-Daten
            
        Returns:
            Verarbeitete Daten oder None bei Fehler
        """
        try:
            # Erwartete Spaltenstruktur
            expected_columns = ['Datum', 'Zeit', 'Lot-Nr.', 'Subsystem', 'Ereignis & Massnahme', 'Visum']
            
            # Spalten bereinigen
            df_cleaned = df.copy()
            df_cleaned.columns = [col.strip() for col in df_cleaned.columns]
            
            # Prüfe ob alle erwarteten Spalten vorhanden sind
            missing_columns = [col for col in expected_columns if col not in df_cleaned.columns]
            if missing_columns:
                logger.warning(f"Fehlende Spalten: {missing_columns}")
                # Füge fehlende Spalten mit N/A hinzu
                for col in missing_columns:
                    df_cleaned[col] = 'N/A'
            
            # Daten bereinigen
            for col in expected_columns:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].astype(str).fillna('').str.strip()
            
            # Leere Ereignis-Einträge entfernen (Hauptinhalt)
            if 'Ereignis & Massnahme' in df_cleaned.columns:
                before_count = len(df_cleaned)
                df_cleaned = df_cleaned[
                    (df_cleaned['Ereignis & Massnahme'].str.strip() != '') & 
                    (df_cleaned['Ereignis & Massnahme'] != 'N/A') &
                    (df_cleaned['Ereignis & Massnahme'] != 'nan')
                ]
                after_count = len(df_cleaned)
                
                if before_count != after_count:
                    logger.info(f"Leere Einträge entfernt: {before_count - after_count}")
            
            # Datum verarbeiten
            if 'Datum' in df_cleaned.columns:
                try:
                    df_cleaned['Datum_Parsed'] = pd.to_datetime(df_cleaned['Datum'], errors='coerce')
                except:
                    logger.warning("Datum-Parsing fehlgeschlagen")
            
            if len(df_cleaned) == 0:
                logger.error("Keine gültigen Daten nach Bereinigung")
                return None
            
            logger.info(f"Datenverarbeitung abgeschlossen: {len(df_cleaned)} gültige Einträge")
            return df_cleaned.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Fehler bei Datenverarbeitung: {e}")
            return None
    
    def create_embeddings(self, df: pd.DataFrame) -> bool:
        """
        Erstellt semantische Embeddings für die Daten.
        
        Args:
            df: Verarbeitete Logbuch-Daten
            
        Returns:
            Erfolg der Embedding-Erstellung
        """
        try:
            logger.info("Erstelle semantische Embeddings...")
            
            texts = []
            for _, row in df.iterrows():
                parts = []
                
                if 'Lot-Nr.' in row and row['Lot-Nr.'] not in ['N/A', 'nan', '']:
                    parts.append(f"Lot: {row['Lot-Nr.']}")
                
                if 'Subsystem' in row and row['Subsystem'] not in ['N/A', 'nan', '']:
                    parts.append(f"System: {row['Subsystem']}")
                
                if 'Ereignis & Massnahme' in row and row['Ereignis & Massnahme'] not in ['N/A', 'nan', '']:
                    ereignis = str(row['Ereignis & Massnahme'])[:200]
                    parts.append(f"Ereignis: {ereignis}")
                
                if 'Datum' in row and row['Datum'] not in ['N/A', 'nan', '']:
                    parts.append(f"Datum: {row['Datum']}")
                
                combined_text = " | ".join(parts) if parts else "Leerer Eintrag"
                texts.append(combined_text)
            
            # Embeddings erstellen
            self.embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # FAISS-Index erstellen
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            self.df = df
            
            logger.info(f"Embeddings erstellt: {len(texts)} Einträge, {dimension} Dimensionen")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Embedding-Erstellung: {e}")
            return False
    
    def semantic_search(self, query: str, max_results: int = 50) -> Tuple[List[int], List[float], int]:
        """
        Führt semantische Suche durch.
        
        Args:
            query: Suchanfrage
            max_results: Maximale Ergebnisanzahl
            
        Returns:
            (indices, scores, optimal_count)
        """
        try:
            if self.index is None or self.model is None:
                return [], [], 0
            
            # Query für E5-Modell optimieren
            optimized_query = f"query: {query}"
            
            # Query-Embedding
            query_embedding = self.model.encode(
                [optimized_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Suche durchführen
            available_count = self.index.ntotal
            actual_max = min(max_results, available_count)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), actual_max)
            
            scores_list = [float(score) for score in scores[0]]
            indices_list = [int(idx) for idx in indices[0]]
            
            # Optimale Anzahl bestimmen
            optimal_count = self._determine_optimal_results(scores_list, query)
            
            return indices_list[:optimal_count], scores_list[:optimal_count], optimal_count
            
        except Exception as e:
            logger.error(f"Fehler bei semantischer Suche: {e}")
            return [], [], 0
    
    def _determine_optimal_results(self, scores: List[float], query: str) -> int:
        """Bestimmt optimale Anzahl der Ergebnisse basierend auf Relevanz."""
        if not scores:
            return 0
        
        scores_array = np.array(scores)
        
        # Relevanz-Schwellenwerte
        high_threshold = 0.65
        medium_threshold = 0.45
        
        high_count = int(np.sum(scores_array >= high_threshold))
        medium_count = int(np.sum(scores_array >= medium_threshold))
        
        # Adaptive Logik
        if high_count >= 3:
            return min(high_count + 3, 20)
        elif medium_count >= 2:
            return min(medium_count + 2, 15)
        else:
            return min(5, len(scores))
    
    def query_ollama(self, prompt: str, context: str) -> str:
        """
        Fragt Ollama LLM für intelligente Analyse ab.
        
        Args:
            prompt: Benutzeranfrage
            context: Gefundener Kontext
            
        Returns:
            LLM-Antwort oder Fallback
        """
        try:
            system_prompt = """Du bist ein Experte für industrielle Logbuch-Analyse.

Analysiere die Daten präzise und beantworte die Frage vollständig.
Strukturiere deine Antwort mit:
1. Direkte Antwort
2. Relevante Einträge
3. Zusammenfassung
4. Empfehlungen

Verwende nur die bereitgestellten Daten."""

            full_prompt = f"""{system_prompt}

KONTEXT:
{context[:3000]}

FRAGE: {prompt}

ANTWORT:"""

            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 800
                }
            }
            
            # Ollama anfragen
            for url in ["http://localhost:11434/api/generate", "http://127.0.0.1:11434/api/generate"]:
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get('response', '').strip()
                        if len(answer) > 10:
                            return answer
                except:
                    continue
            
            # Fallback
            return self._fallback_analysis(prompt, context)
                
        except Exception as e:
            logger.error(f"Ollama-Fehler: {e}")
            return self._fallback_analysis(prompt, context)
    
    def _fallback_analysis(self, prompt: str, context: str) -> str:
        """Fallback-Analyse ohne LLM."""
        lines = context.split('\n')
        entry_count = len([line for line in lines if 'Eintrag' in line])
        
        return f"""**Basis-Analyse** (KI nicht verfügbar)

**Antwort:** {entry_count} relevante Einträge zu Ihrer Anfrage gefunden.

**Kontext:**
{context[:600]}...

**Hinweis:** Für detaillierte KI-Analyse Ollama starten: `ollama serve`"""
    
    def analyze_query(self, query: str) -> Dict:
        """
        Hauptanalysefunktion - kombiniert Suche und KI-Analyse.
        
        Args:
            query: Benutzeranfrage
            
        Returns:
            Vollständige Analyseergebnisse
        """
        if self.df is None or self.index is None:
            return {"error": "Keine Daten geladen"}
        
        try:
            logger.info(f"Analysiere: '{query}'")
            
            # Semantische Suche
            indices, scores, result_count = self.semantic_search(query)
            
            if not indices:
                return {"error": "Keine relevanten Ergebnisse gefunden"}
            
            # Relevante Einträge extrahieren
            relevant_entries = self.df.iloc[indices].copy()
            relevant_entries['similarity_score'] = scores
            
            # Kontext für LLM aufbereiten
            context_parts = []
            for i, (_, row) in enumerate(relevant_entries.iterrows(), 1):
                entry_text = f"""Eintrag {i}:
- Datum: {row.get('Datum', 'N/A')}
- Zeit: {row.get('Zeit', 'N/A')}
- Lot-Nr.: {row.get('Lot-Nr.', 'N/A')}
- Subsystem: {row.get('Subsystem', 'N/A')}
- Ereignis: {row.get('Ereignis & Massnahme', 'N/A')}
- Visum: {row.get('Visum', 'N/A')}
- Relevanz: {scores[i-1]:.3f}
"""
                context_parts.append(entry_text)
            
            context = "\n".join(context_parts)
            
            # Ergebnisse zusammenstellen
            result = {
                "relevant_entries": relevant_entries,
                "context": context,
                "query": query,
                "result_count": result_count,
                "total_available": len(self.df),
                "average_relevance": float(np.mean(scores)),
                "max_relevance": float(max(scores)) if scores else 0,
                "search_quality": "Hoch" if np.mean(scores) > 0.6 else "Mittel" if np.mean(scores) > 0.4 else "Niedrig"
            }
            
            # KI-Analyse
            logger.info("Starte KI-Analyse...")
            result["llm_analysis"] = self.query_ollama(query, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysefehler: {e}")
            return {"error": f"Analysefehler: {str(e)}"}

def load_and_setup_analyzer(csv_file_path: str = "Logbuch_Spritzgussanlage_NEU.csv") -> Optional[LogbookAnalyzer]:
    """
    Lädt und konfiguriert den Analyzer mit echten Daten.
    
    Args:
        csv_file_path: Pfad zur CSV-Datei
        
    Returns:
        Konfigurierter Analyzer oder None bei Fehler
    """
    print(f"Lade und konfiguriere Logbook Analyzer...")
    
    if not os.path.exists(csv_file_path):
        print(f"Fehler: CSV-Datei nicht gefunden: {csv_file_path}")
        return None
    
    # Analyzer initialisieren
    analyzer = LogbookAnalyzer()
    
    # Embedding-Modell laden
    if not analyzer.load_embedding_model():
        print("Fehler: Embedding-Modell konnte nicht geladen werden")
        return None
    
    # CSV-Daten laden
    df = analyzer.load_csv_data(csv_file_path)
    if df is None:
        print("Fehler: CSV-Daten konnten nicht geladen werden")
        return None
    
    # Daten verarbeiten
    processed_df = analyzer.preprocess_data(df)
    if processed_df is None:
        print("Fehler: Datenverarbeitung fehlgeschlagen")
        return None
    
    # Embeddings erstellen
    if not analyzer.create_embeddings(processed_df):
        print("Fehler: Embedding-Erstellung fehlgeschlagen")
        return None
    
    print(f"Analyzer erfolgreich konfiguriert:")
    print(f"  - Dateneinträge: {len(processed_df)}")
    print(f"  - Embedding-Modell: {analyzer.model_name}")
    print(f"  - Bereit für Analyse")
    
    return analyzer

def get_data_statistics(analyzer: LogbookAnalyzer) -> Dict:
    """
    Erstellt Statistiken über die geladenen Daten.
    
    Args:
        analyzer: Konfigurierter LogbookAnalyzer
        
    Returns:
        Dictionary mit Datenstatistiken
    """
    if analyzer.df is None:
        return {"error": "Keine Daten verfügbar"}
    
    df = analyzer.df
    stats = {
        "total_entries": len(df),
        "columns": list(df.columns),
        "date_range": None,
        "subsystem_distribution": {},
        "event_text_stats": {}
    }
    
    # Zeitraum-Analyse
    if 'Datum_Parsed' in df.columns:
        valid_dates = df['Datum_Parsed'].dropna()
        if len(valid_dates) > 0:
            stats["date_range"] = {
                "start": str(valid_dates.min().date()),
                "end": str(valid_dates.max().date()),
                "days": (valid_dates.max() - valid_dates.min()).days
            }
    
    # Subsystem-Verteilung
    if 'Subsystem' in df.columns:
        subsystem_counts = df['Subsystem'].value_counts()
        stats["subsystem_distribution"] = subsystem_counts.head(10).to_dict()
    
    # Ereignis-Text-Statistiken
    if 'Ereignis & Massnahme' in df.columns:
        text_lengths = df['Ereignis & Massnahme'].str.len()
        stats["event_text_stats"] = {
            "avg_length": float(text_lengths.mean()),
            "median_length": float(text_lengths.median()),
            "min_length": int(text_lengths.min()),
            "max_length": int(text_lengths.max())
        }
    
    return stats