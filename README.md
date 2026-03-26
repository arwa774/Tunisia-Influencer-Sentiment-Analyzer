# Tunisian-Influencer-Sentiment-Scoring-System

Un pipeline complet pour scraper les commentaires des influenceurs tunisiens sur **TikTok** et **YouTube**, classifier le sentiment de chaque commentaire avec **Qwen2.5-7B-Instruct**, et attribuer un score à chaque influenceur basé sur la distribution des sentiments de son audience — incluant un **score global** qui fusionne les deux plateformes.

---

## Apercu du Projet

Ce projet cible les créateurs de contenu tunisiens sur TikTok et YouTube. Il collecte les commentaires de leur audience à grande échelle, les passe dans un classifieur basé sur un LLM qui comprend le **Darja tunisien** (un dialecte qui mélange arabe, français et anglais), et produit un score par influenceur — par plateforme individuellement et en score global combiné.

### Résumé du Pipeline

```
Liste des influenceurs (CSV)
        |
        v
+-------------------+     +-------------------+
|  YouTube Scraper  |     |  TikTok Scraper   |
|  (YouTube API v3) |     |  (Async HTTP +    |
|                   |     |   yt-dlp)         |
+--------+----------+     +--------+----------+
         |                         |
         +-----------+-------------+
                     |
                     v
          CSV brut des commentaires
                     |
                     v
        +----------------------------------+
        |        Nettoyage des données     |
        |  - Suppression des valeurs nulles|
        |  - Retrait des users < 9 comments|
        |  - Limite de 300 comments/user   |
        +---------------+------------------+
                        |
                        v
        +----------------------------------+
        |     Classifieur de sentiment     |
        |     Qwen2.5-7B-Instruct          |
        |   + Emoji Lexicon Boost          |
        |   + Darja-aware System Prompt    |
        +---------------+------------------+
                        |
                        v
        +----------------------------------+
        |      Scoring des influenceurs    |
        |  - Score par plateforme          |
        |    (pondéré par confidence,      |
        |     de -100 a +100)              |
        |  - Score global (YT + TT fusionné|
        +----------------------------------+
```

---

## Structure du Dépôt

```
scraping/
    youtube_scraper.py        # Scrape les commentaires via YouTube Data API v3 (rotation de clés + checkpoint)
    tiktok_main.py            # Boucle principale de scraping TikTok (par user, reprise automatique)
    scrap_tiktok.py           # Fetcher asynchrone de commentaires TikTok (anti-détection, support proxy)
    url_finder.py             # Extrait toutes les URLs de vidéos d'un profil TikTok via yt-dlp

model/
    sentiment_classifier.py   # Analyse de sentiment avec Qwen2.5-7B (batch inference, emoji boost)

notebooks/
    cleaning_and_scoring.ipynb  # Nettoyage des données + scoring par plateforme + fusion globale

requirements.txt
README.md
```

---

## Fonctionnalités

### Scraping

- **YouTube** : utilise YouTube Data API v3 avec rotation automatique des clés en cas d'épuisement du quota. Collecte entre 100 et 1000 commentaires par chaîne à travers plusieurs vidéos, avec un système de checkpoint et reprise automatique.
- **TikTok** : scraping HTTP asynchrone avec rotation du user-agent, support proxy, et délais aléatoires pour simuler un comportement humain et éviter la détection. Les URLs des vidéos sont extraites via `yt-dlp`. Système de checkpoint identique.

### Nettoyage des données (`cleaning_and_scoring.ipynb`)

Le notebook applique les étapes suivantes sur les CSV bruts :

1. **Suppression des lignes nulles** — supprime tout commentaire avec une valeur manquante
2. **Filtrage des utilisateurs à faible volume** — retire les influenceurs avec moins de 9 commentaires (signal insuffisant)
3. **Limite par utilisateur à 300 commentaires** — échantillonne aléatoirement 300 commentaires pour les utilisateurs qui en ont plus, afin qu'aucun influenceur ne domine le dataset

Produit les fichiers : `cleaned_youtube_comments.csv` et `cleaned_tiktok_comments.csv` (ainsi que les variantes `final_cleaned_*` prêtes pour le classifieur).

### Classification des sentiments

- Modèle : `Qwen/Qwen2.5-7B-Instruct` chargé en quantification **8-bit** via BitsAndBytes
- **Batch inference** : 20 commentaires par appel au LLM (~10 à 15 fois plus rapide qu'un par un)
- **System prompt adapté au Darja** : inclut des exemples few-shot et un glossaire de termes du dialecte tunisien (ex. `wallah`, `barcha`, `3ajib`, `ma7la`, `khayeb`, `nti7a`)
- **Emoji lexicon** : plus de 150 emojis associés à des tuples de scores `(positif, négatif, neutre)`
- **Emoji hybrid boost** : pour les commentaires mixtes texte+emoji, les signaux emoji peuvent surpasser ou renforcer la prédiction du LLM
- **Fast path emoji-only** : les commentaires composés uniquement d'emojis sont classifiés directement via le lexique, sans utiliser le GPU
- **Sauvegarde checkpoint** toutes les 100 commentaires — résistant aux crashes et reprise possible

### Scoring des influenceurs (`cleaning_and_scoring.ipynb`)

#### Score par plateforme

Chaque influenceur reçoit un **Score de Sentiment Pondéré par la Confidence** par plateforme :

```
score_i    = confidence_i x sentiment_value_i

sentiment_value :  +1 (Positive)  |  0 (Neutral)  |  -1 (Negative)

Score utilisateur = ( somme(score_i) / n ) x 100
```

| Score | Signification |
|---|---|
| `+100` | Tous les commentaires positifs avec 100% de confidence |
| `-100` | Tous les commentaires négatifs avec 100% de confidence |
| `0` | Parfaitement équilibré ou tout neutre |

Produit : `user_sentiment_summary_youtube.csv` et `user_sentiment_summary_tiktok.csv`

#### Score global (fusion cross-plateforme)

Le notebook fusionne ensuite les deux résumés en un seul fichier via une jointure externe :

- **Utilisateur présent sur les deux plateformes** : `global_score = (youtube_score + tiktok_score) / 2`
- **Utilisateur sur une seule plateforme** : `global_score = score de la plateforme disponible`
- Les données manquantes sont remplies par `0` et détectées via `total_comments == 0`

Produit : `user_sentiment_global.csv`

---

## Démarrage rapide

### 1. Cloner le dépôt

```bash
git clone https://github.com/arwa774/Tunisia-Influencer-Sentiment-Analyzer.git
cd Tunisia-Influencer-Sentiment-Analyzer
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
API_KEYS=YOUR_YOUTUBE_API_KEY_1,YOUR_YOUTUBE_API_KEY_2
```

### 4. Préparer la liste des influenceurs

Le fichier CSV d'entrée (`dataset_ml.csv`) doit contenir une colonne `username` avec les pseudos TikTok/YouTube.

### 5. Lancer les scrapers

```bash
# YouTube
python scraping/youtube_scraper.py

# TikTok
python scraping/tiktok_main.py
```

### 6. Nettoyer les données

Ouvrir `notebooks/cleaning_and_scoring.ipynb` dans **Google Colab** et exécuter la section de nettoyage. Cela génèrera :
- La suppression des valeurs nulles
- Le filtrage des influenceurs avec moins de 9 commentaires
- La limite à 300 commentaires par influenceur
- Les fichiers `final_cleaned_youtube_comments.csv` et `final_cleaned_tiktok_comments.csv`

### 7. Lancer le classifieur de sentiment

Conçu pour s'exécuter sur GPU (Kaggle ou Google Colab recommandé). Mettre à jour le chemin `DEFAULT_INPUT` dans le script pour pointer vers le CSV nettoyé :

```bash
python model/sentiment_classifier.py
```

### 8. Scorer les influenceurs

De retour dans `cleaning_and_scoring.ipynb`, exécuter la section de scoring. Cela génèrera :
- Les scores de sentiment pondérés par plateforme
- La fusion des scores YouTube et TikTok en un classement global
- Le fichier `user_sentiment_global.csv`

---

## Dépendances

Dépendances principales (voir `requirements.txt` pour la liste complète) :

| Package | Utilité |
|---|---|
| `google-api-python-client` | YouTube Data API v3 |
| `yt-dlp` | Extraction des URLs de vidéos TikTok |
| `httpx` | HTTP asynchrone pour le scraping TikTok |
| `transformers` | Chargement du modèle Qwen2.5-7B |
| `bitsandbytes` | Quantification 8-bit du modèle |
| `torch` | Inférence GPU |
| `pandas` | Manipulation des données |
| `tqdm` | Suivi de la progression |
| `python-dotenv` | Gestion des clés API |

---

## Fichiers de sortie

| Fichier | Description |
|---|---|
| `tunisian_youtubers_comments.csv` | Commentaires YouTube bruts scrapés |
| `tiktok_comments.csv` | Commentaires TikTok bruts scrapés |
| `cleaned_youtube_comments.csv` | Commentaires YouTube nettoyés (nulls supprimés, utilisateurs à faible volume retirés) |
| `final_cleaned_youtube_comments.csv` | Commentaires YouTube limités à 300 par utilisateur — prêts pour le classifieur |
| `result_youtube.csv` | Commentaires YouTube avec sentiment prédit |
| `result_tiktok.csv` | Commentaires TikTok avec sentiment prédit |
| `user_sentiment_summary_youtube.csv` | Statistiques de sentiment + score par influenceur (YouTube) |
| `user_sentiment_summary_tiktok.csv` | Statistiques de sentiment + score par influenceur (TikTok) |
| `user_sentiment_global.csv` | **Fichier final** — score global fusionné par influenceur |

### Colonnes de sortie du classifieur

| Colonne | Description |
|---|---|
| `predicted_sentiment` | `Positive`, `Negative` ou `Neutral` |
| `confidence` | Score entre 0 et 1 |
| `prediction_source` | `hf_llm_batch`, `emoji_lexicon` ou `hf_llm_batch+emoji_override` |

### Colonnes de sortie du score global

| Colonne | Description |
|---|---|
| `user_name` | Pseudo de l'influenceur |
| `youtube_*` | Toutes les statistiques et le score YouTube |
| `tiktok_*` | Toutes les statistiques et le score TikTok |
| `global_score` | Score de sentiment cross-plateforme final (-100 à +100) |

---

## Notes importantes

- Un **proxy** est fortement recommandé pour le scraping TikTok afin d'éviter les blocages IP.
- Le classifieur de sentiment nécessite un **GPU** avec au moins 16 Go de VRAM (Kaggle T4 / A100 fonctionne bien).
- Les clés API YouTube ont une limite de quota quotidienne — le scraper effectue une rotation automatique entre plusieurs clés.
- La structure de l'API TikTok peut évoluer ; le scraper extrait dynamiquement les tokens (`msToken`, `aid`, `region`) depuis le HTML de la page vidéo à chaque exécution.
