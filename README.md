# Pipeline Data & IA — De la facture scannée à l'Agent conversationnel

**Auteurs :** Théo Constantin & Quentin Jacquot  

Chaîne de traitement de données reproductible de bout en bout (**Data Engineering + IA Agentique**) : ingestion incrémentale de factures PDF depuis S3, OCR par **LightOnOCR-3**, architecture Médaillon **Bronze / Silver / Gold** en **Apache Parquet**, entrepôt analytique **DuckDB**, serveur **MCP (Model Context Protocol)** en lecture seule et **Agent IA multi-LLM** en français.

---

## Architecture globale

```text
PDF (S3 Hyperfluid)
   │
   ▼  Étape 1 : PyMuPDF (détection texte vs scan)
data/raw_pdfs/
   │
   ├──► PDF natifs (50) ──► PyMuPDF texte ──┐
   │                                        ▼  Étape 3 (SHA-256, tolérance pannes)
   └──► PDF scannés (50) ─► LightOnOCR-3 ──► data/bronze/documents.parquet
                            (Étape 2)             │
                                                  ▼  Étape 4 (Decimal, qualité, quarantaine)
                                            data/silver/
                                              ├── factures.parquet (99 valides)
                                              ├── lignes_facture.parquet (270 lignes)
                                              └── quarantaine.parquet (1 rejetée)
                                                  │
                                                  ▼  Étape 6 (vues SQL relatives, 0 table physique)
                                            data/warehouse.duckdb (bronze / silver / gold)
                                                  │
                                                  ▼  Étape 7 (FastMCP stdio, read_only=True)
                                            src/mcp_server.py
                                                  │
                                                  ▼  Étape 8 (Agent multi-LLM configurable)
                                            src/step8_agent.py (gemma-4-12b / granite-4-2-8b / ...)
```

### Structure du dépôt

```text
.
├── .env.example                 # Modèle des variables d'environnement (S3, OCR, LLM)
├── .gitignore                   # Exclut .env, données locales (sauf data/ground_truth.json), .venv
├── pyproject.toml               # Dépendances Python gérées par uv
├── uv.lock                      # Fichier de verrouillage reproductible
├── README.md                    # Guide d'installation et de lancement
├── RAPPORT.md                   # Rapport technique (architecture, benchmark OCR, MCP/Agent, prod)
├── data/
│   └── ground_truth.json        # Vérité terrain stratifiée (15 factures scannées, versionnée)
└── src/
    ├── config.py                # Chargement .env, chemins et résolution des fournisseurs S3/OCR/LLM
    ├── ocr.py                   # Abstraction OCRProvider (ABC) + LightOnOCRProvider
    ├── mcp_client.py            # Client MCP asynchrone (stdio)
    ├── mcp_server.py            # Serveur FastMCP (6 outils en lecture seule, transparence SQL)
    ├── agent/
    │   ├── llm.py               # Connecteurs LLM interchangeables (Hyperfluid, OpenAI-compat, Claude, Gemini)
    │   ├── tools.py             # ToolManager MCP -> LLM
    │   └── chat.py              # Boucle agentique multi-tours et system prompt anti-hallucination
    ├── step1_ingest.py          # Étape 1 : Sync S3 incrémentale + détection texte vs image
    ├── step2_ocr.py             # Étape 2 : Test LightOnOCR-3 sur 1 page + mesure latence
    ├── step3_bronze.py          # Étape 3 : Couche Bronze Parquet (SHA-256, checkpointing, erreurs)
    ├── step4_silver.py          # Étape 4 : Couche Silver typée (Decimal/Date), qualité & quarantaine
    ├── step5_eval_ocr.py        # Étape 5 : Benchmark OCR champ par champ (1540 px vs 512 px)
    ├── step6_warehouse.py       # Étape 6 : Vues DuckDB (bronze/silver/gold) + test concurrence
    ├── step7_test_mcp.py        # Étape 7 : Suite automatisée (13 tests outils + anti-SQLi + read-only)
    └── step8_agent.py           # Étape 8 : CLI Agent (interactif, --query, --compare-models)
```

---

## Prérequis et installation

- **Python** `>= 3.12`
- **[uv](https://docs.astral.sh/uv/)** (`>= 0.5`)

Cloner le dépôt puis installer les dépendances dans l'environnement virtuel géré par `uv` :

```bash
uv sync
```

---

## Configuration (`.env`)

Créer le fichier `.env` à partir du modèle fourni :

```bash
cp .env.example .env
```

Renseigner vos identifiants dans `.env` (ce fichier est ignoré par Git via `.gitignore`) :

| Variable | Rôle | Exemple / Défaut |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Clé d'accès S3 Hyperfluid (lecture seule sur `factures`) | *(à renseigner)* |
| `AWS_SECRET_ACCESS_KEY` | Secret S3 Hyperfluid | *(à renseigner)* |
| `AWS_ENDPOINT_URL` | Point d'accès S3 | `https://dams-5.s3-gateway.console.hyperfluid.cloud` |
| `AWS_DEFAULT_REGION` | Région S3 | `default` |
| `S3_BUCKET_NAME` | Nom du bucket source | `factures` |
| `HYPERFLUID_API_KEY` | Clé API AI Gateway Sapience (OCR + LLMs Hyperfluid) | *(à renseigner)* |
| `OCR_MODEL` | Modèle OCR | `lightonocr-3-1b-test` |
| `OCR_ENDPOINT_URL` | URL vLLM OpenAI-compatible de l'OCR | `https://llm.hyperfluid.cloud/v1/...` |
| `OCR_TARGET_LONGEST_EDGE` | Résolution cible du plus grand côté (px) | `1540` |
| `LLM_PROVIDER` | Fournisseur LLM de l'agent (`hyperfluid`, `openai_compatible`, `claude`, `google`) | `hyperfluid` |
| `LLM_MODEL` | Modèle utilisé par l'agent | `gemma-4-12b` |
| `LLM_BASE_URL` | Optionnel pour `openai_compatible` (ex: `http://localhost:11434/v1`) | *(vide par défaut)* |
| `LLM_API_KEY` | Optionnel (sinon réutilise `HYPERFLUID_API_KEY`) | *(vide par défaut)* |

---

## Lancer le pipeline étape par étape

### Étape 1 — Récupération S3 et détection texte vs image
```bash
# Test rapide sur 5 fichiers
uv run python -m src.step1_ingest --limit 5

# Récupération complète des 100 factures (incrémentale)
uv run python -m src.step1_ingest
```
Produit `data/raw_pdfs/` et `data/raw_pdfs/manifest_step1.json`.

### Étape 2 — Test unitaire du modèle LightOnOCR-3
```bash
# Envoie la 1ère facture scannée à LightOnOCR-3 (1540 px) et compare avec un PDF natif
uv run python -m src.step2_ocr --compare-native
```

### Étape 3 — Couche Bronze en Parquet (incrémentale + tolérance aux pannes)
```bash
# Test de tolérance aux pannes (simule une coupure OCR sur 2 fichiers)
uv run python -m src.step3_bronze --limit 2 --simulate-ocr-error

# Ingestion complète (natifs via PyMuPDF + scans via LightOnOCR-3)
uv run python -m src.step3_bronze

# Vérification d'idempotence : la 2e exécution ignore les 100 documents (0 appel OCR)
uv run python -m src.step3_bronze
```
Produit `data/bronze/documents.parquet` et `data/bronze/ingestion_errors.jsonl`.

### Étape 4 — Couche Silver, contrôles qualité et quarantaine
```bash
uv run python -m src.step4_silver
```
Produit `data/silver/factures.parquet` (99 factures valides), `data/silver/lignes_facture.parquet` (270 lignes) et `data/silver/quarantaine.parquet` (1 facture rejetée : `V-2025-0001`).

### Étape 5 — Évaluation de la qualité OCR (`1540 px` vs `512 px`)
```bash
# Compare les résolutions 1540 px (réutilisée depuis Bronze) et 512 px sur data/ground_truth.json
uv run python -m src.step5_eval_ocr

# Forcer le recalcul sans cache
uv run python -m src.step5_eval_ocr --no-cache
```
Exporte les résultats détaillés dans `data/eval_ocr_results.json`.

### Étape 6 — Entrepôt DuckDB (séparation stockage / calcul)
```bash
# Crée les vues (bronze, silver, gold, main) et lance le test de concurrence multi-processus
uv run python -m src.step6_warehouse

# Exécuter une requête SQL ponctuelle en lecture seule
uv run python -m src.step6_warehouse --sql "SELECT status, COUNT(*), SUM(total_ttc) FROM silver.factures GROUP BY status"
```

### Étape 7 — Validation automatisée du serveur MCP
```bash
uv run python -m src.step7_test_mcp
```
Lance `src/mcp_server.py` en sous-processus `stdio` via `MCPClient` et exécute **13 tests automatisés** (découverte des 6 outils, schéma, totaux 2025, neutralisation d'injection SQL, quarantaine `V-2025-0001`, couche Gold, auto-correction par `hint`, blocage de `DROP`/`DELETE`/`CREATE`).

### Étape 8 — Agent IA Multi-LLM
```bash
# Mode interactif (conversation multi-tours, 'quit' pour sortir)
uv run python -m src.step8_agent

# Question unique ou suivie en CLI
uv run python -m src.step8_agent --query "Quel est le montant TTC des factures en retard en 2025 ?"

# Changer de LLM à la volée (ex: granite-4-2-8b ou modèle local Ollama)
uv run python -m src.step8_agent --model granite-4-2-8b --query "Quel est le montant TTC des factures en retard en 2025 ?"

# Benchmark comparatif multi-LLM avec vérification SQL automatique
uv run python -m src.step8_agent --compare-models --models "gemma-4-12b,granite-4-2-8b"
```

---

## Consultation depuis DBeaver (accès concurrent sans blocage)

L'entrepôt `data/warehouse.duckdb` ne contient aucune table physique (uniquement des vues SQL relatives sur Parquet). Pour l'explorer depuis **DBeaver** pendant que le serveur MCP ou Python tourne :

1. Créer une nouvelle connexion **DuckDB** pointant vers :
   `<chemin-absolu-du-projet>/data/warehouse.duckdb`
2. Dans **Driver Properties** (propriétés JDBC), ajouter ou définir :
   `duckdb.read_only` = `true`
3. Vérifier que le répertoire de travail de DBeaver permet la résolution des chemins relatifs `data/...`, ou exécuter `SET file_search_path = '<chemin-absolu-du-projet>';` en début de session SQL.

---

## Intégration MCP dans Claude Desktop / Claude Code

Ajouter la configuration suivante dans le fichier de configuration MCP (`claude_desktop_config.json` ou `.mcp.json`) en adaptant le chemin du dépôt :

```json
{
  "mcpServers": {
    "factures-warehouse": {
      "command": "uv",
      "args": [
        "--directory",
        "/chemin/absolu/vers/tp-data-ia-factures",
        "run",
        "python",
        "-m",
        "src.mcp_server"
      ]
    }
  }
}
```
