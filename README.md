# SP Moodboard Generator

SP Moodboard Generator est une application locale pour construire des moodboards Bento, analyser leur coherence esthetique et explorer les relations latentes entre images, couleurs, objets, symboles, affects, styles et compositions.

L'app propose deux vues principales:

- **Bento**: generation de grilles editorialisees, export PNG/PDF, pages d'analyse ajoutees au PDF.
- **Graph**: graphe multimodal interactif ou chaque image et chaque modalite detectee devient un noeud relie par des similarites, co-occurrences et affinites esthetiques.

![Screenshot de l'application](screenshot.png)

Le projet est pense comme un MVP local de "moodboard cognitif": il ne se contente pas de dire si des images se ressemblent visuellement, il cherche a mesurer si elles forment une harmonie latente, quels regimes esthetiques coexistent, quels elements renforcent ou cassent la purete, et comment les images se regroupent en clusters.

## Fonctionnalites

- Upload multi-images.
- Generation Bento avec modes `grid`, `random`, `custom`, `auto`, `simple`.
- Bento optimizer optionnel, desactive par defaut, avec modes `Balanced`, `Editorial`, `Dense`, `Clustered`.
- Analyse locale via `/api/analyze`.
- Graphe interactif avec noeuds `image`, `color`, `object`, `symbol`, `texture`, `style`, `emotion`, `affect`, `composition`, `aesthetic`.
- Clustering latent et couleurs de clusters.
- Palette globale, couleurs nommees localement, coherence colorimetrique.
- Matching avec une base locale d'esthetiques.
- Analyse spectrale du graphe avec Laplacien.
- Export PDF/PNG/ZIP avec `analysis.json`.
- Pages PDF d'analyse avec resume, palette, tags, nuage de points, clusters, spectral analysis et scores.
- Cache memoire pour les previews, analyses et modeles charges.

## Lancer L'Application

### 1. Creer l'environnement Python

Le projet cible Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

### 2. Lancer le serveur local

Mode rapide recommande pour tester l'interface:

```bash
MOODBOARD_CAPTION_BACKEND=fast MOODBOARD_ENABLE_OWLV2=0 .venv/bin/python app/server.py --port 8787
```

Puis ouvrir:

```text
http://127.0.0.1:8787
```

Le serveur expose aussi:

- `GET /api/health`
- `GET /api/models`
- `GET /api/aesthetics`
- `POST /api/preview`
- `POST /api/analyze`
- `POST /api/generate`

### 3. Variables utiles

```bash
MOODBOARD_ENABLE_ML=0
```

Desactive les embeddings ML et force les fallbacks heuristiques.

```bash
MOODBOARD_CAPTION_BACKEND=fast
```

Utilise le backend caption rapide si le modele local est disponible.

```bash
MOODBOARD_CAPTION_BACKEND=florence
```

Utilise Florence-2 pour captions et regions, plus lent mais plus riche.

```bash
MOODBOARD_ENABLE_OWLV2=0
```

Desactive le grounding open-vocabulary OWLv2 pour accelerer l'analyse.

## Modeles Locaux

L'application est configuree pour travailler localement/offline. Les modeles Hugging Face sont charges depuis `data/huggingface` avec `local_files_only=True`. Si un modele manque, le pipeline degrade proprement vers des fallbacks heuristiques quand c'est possible.

Modeles et roles principaux:

| Role | Modele actif ou recommande | Usage |
| --- | --- | --- |
| Embeddings multimodaux | `google/siglip2-base-patch16-224` | embeddings image/texte, zero-shot, aesthetic matching |
| Attention | SigLIP2 patch tokens | signature de salience visuelle |
| Caption rapide | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | captions rapides |
| Caption/regions | `microsoft/Florence-2-base` | captions, regions, concepts |
| Detection ouverte | `google/owlv2-base-patch16-ensemble` | objets/symboles open-vocabulary, optionnel |
| Future backbone visuel | `facebook/dinov2-base` | style, silhouette, retrieval visuel |
| Future segmentation | `facebook/sam2-hiera-small` | masques, silhouette, foreground |

Le statut reel des modeles locaux est visible dans:

```text
http://127.0.0.1:8787/api/models
```

## Arborescence

```text
.
├── app/
│   ├── server.py                  # serveur HTTP local stdlib
│   └── api/
│       ├── analyze.py             # endpoint /api/analyze
│       ├── generate.py            # preview/export PDF PNG ZIP
│       ├── aesthetics.py          # endpoint base esthetique
│       └── models.py              # endpoint statut modeles
├── frontend/
│   ├── moodboard_interface.html   # interface principale
│   └── static/
│       ├── css/moodboard.css
│       └── js/
│           ├── bento_view.js
│           ├── graph_view.js
│           └── spectral_view.js
├── src/moodboard/
│   ├── core/
│   │   ├── schemas.py             # normalisation params/API
│   │   ├── paths.py               # chemins projet/database/data
│   │   ├── cache.py               # caches memoire preview/analyse/modeles
│   │   └── image_io.py            # ouverture images, metadonnees visuelles
│   ├── bento/
│   │   ├── layout.py              # templates et placement Bento
│   │   ├── optimizer.py           # scoring et plans Bento optimises
│   │   ├── render.py              # rendu PIL des pages
│   │   └── pdf_export.py          # PDF, PNG, ZIP, pages d'analyse
│   ├── analysis/
│   │   ├── moodboard_analyzer.py  # pipeline IA principal
│   │   ├── graph_builder.py       # construction du graphe multimodal
│   │   ├── clustering.py          # HDBSCAN/KMeans + projection UMAP
│   │   ├── purity.py              # calcul des scores de purete
│   │   ├── spectral.py            # analyse spectrale Laplacienne
│   │   └── scoring.py             # heuristiques visuelles de secours
│   └── ai/
│       ├── orchestrator.py        # selection des backends IA
│       ├── registry.py            # registre modeles/artifacts
│       ├── models/                # adapters embeddings, captions, colors, world
│       └── variables/             # taxonomies, labels, prompts, modalites
├── scraping/
│   ├── refresh_database.py        # CLI de refresh database
│   ├── aesthetic_sources.py       # Aesthetics Wiki + CARI metadata-only
│   └── scrape_color_names.py      # base locale de noms de couleurs
├── database/
│   ├── aesthetics_cache.json      # cache local esthetiques
│   ├── color_names.json           # noms de couleurs locaux
│   ├── dataset_registry.json
│   ├── model_registry.json
│   └── world_model_index.json
├── data/
│   ├── huggingface/               # cache local des modeles HF
│   └── trained_models/            # artifacts produits offline
├── world_learning/
│   ├── data_factory.py            # normalisation datasets vers schema commun
│   ├── train_world_model.py       # entrainement/index world model
│   ├── train_fusion_encoder.py    # calibration/fusion multimodale
│   ├── calibrate.py
│   ├── evaluate.py
│   └── dataset_adapters/          # LAION, AVA, BAM, BAID, Polyvore, Danbooru
├── configs/
│   ├── models/                    # configs par modele IA
│   ├── pipelines/                 # fast/balanced/deep
│   ├── training/                  # configs d'apprentissage
│   └── variables/                 # poids, clustering, purity, spectral
├── datasets/                      # datasets externes ou futurs corpus
├── dataset-test-1/                # jeux d'essai locaux
├── dataset-test-2/
├── dataset-test-3/
├── moodboard_app.py               # facade compatibilite legacy
├── requirements-ml.txt
└── screenshot.png
```

## Pipeline D'Analyse

Le pipeline principal vit dans `src/moodboard/analysis/moodboard_analyzer.py`.

### 1. Normalisation des entrees

Le serveur parse les uploads, nettoie les noms de fichiers et normalise les parametres via `normalize_params`.

Chaque image devient un `UploadedImage`, puis un `ImageInfo` contenant:

- largeur, hauteur, orientation;
- aire;
- couleur moyenne HSV;
- luminosite;
- contraste;
- score hero;
- teinte d'accent.

Ces informations sont rapides a calculer et servent a la fois au Bento, aux fallbacks et aux premiers scores.

### 2. Palette et couleurs

Pour chaque image, le pipeline extrait une palette locale via `image_palette`:

- image reduite;
- quantization MedianCut;
- deduplication perceptuelle en Lab;
- nommage via `database/color_names.json`;
- roles `dominant`, `secondary`, `accent`;
- poids relatif de chaque couleur.

La palette globale fusionne les couleurs similaires par cle canonique. Les repetitions de couleurs comme `black` ou `lavender` deviennent donc un seul noeud couleur partage dans le graphe.

La coherence couleur est calculee avec:

- compacite Lab autour de la moyenne ponderee;
- concentration des couleurs dominantes;
- penalite de longue traine si la palette est trop dispersee.

### 3. Embeddings Visuels

Le modele principal est SigLIP2:

```text
google/siglip2-base-patch16-224
```

Il produit:

- un embedding image global;
- des embeddings texte pour les modalites;
- des similarites zero-shot;
- une signature d'attention/salience a partir des patch tokens.

Si SigLIP2 n'est pas disponible localement, le systeme utilise un embedding heuristique construit depuis:

- orientation;
- palette;
- luminosite;
- contraste;
- densite d'edges;
- score hero.

### 4. Captions, Objets, Symboles Et Modalites

Selon `MOODBOARD_CAPTION_BACKEND`, le pipeline peut utiliser:

- SmolVLM2 pour captions rapides;
- Florence-2 pour captions plus riches et regions;
- OWLv2 pour grounding open-vocabulary si active.

Les sorties libres sont converties en observations typées:

- `color`
- `object`
- `symbol`
- `texture`
- `style`
- `emotion`
- `affect`
- `composition`
- `tag`

Les observations portent une confiance, une source et parfois des metadata comme bounding boxes ou valeurs RGB.

Le point important: l'interface ne doit pas inventer ces modalites. Elle lit les observations produites par les modeles et les adapters.

### 5. Fusion Multimodale

L'app ne travaille pas uniquement avec un embedding image brut. Elle construit un vecteur esthetique composite.

Pour chaque image:

```text
V_image =
  visual
  + colors
  + objects
  + symbols
  + textures
  + styles
  + emotions
  + affects
  + composition
  + tags
```

Chaque modalite est transformee en phrase courte, puis embeddee par la tour texte SigLIP2. Exemple:

```text
palette: lavender, black, cold blue
object: cathedral, armor, window
symbolic affective values: mortality, heroism, transgression
composition: central balance, negative space
```

Ces vecteurs texte sont fusionnes avec l'embedding visuel selon des poids calibrables. Les poids par defaut sont dans `fusion_encoder.py`, et peuvent etre ajustes par l'artifact:

```text
data/trained_models/fusion_calibrator_v1.json
```

Le resultat est un **unified aesthetic embedding** utilise pour:

- clustering;
- graph layout;
- purete;
- spectral analysis;
- matching world model.

### 6. Clustering

Les embeddings composites sont normalises puis clusters.

Strategie:

1. HDBSCAN sur distance cosine pre-calculee.
2. Si HDBSCAN ne trouve pas de regimes stables, fallback KMeans.
3. KMeans choisit le nombre de clusters par silhouette cosine, avec une penalite douce si le nombre de clusters s'eloigne trop de la cible.
4. Les outliers sont marques si leur similarite moyenne aux autres images est trop basse.

Chaque cluster reçoit une couleur stable:

```text
#5D71FC, #EB5757, #f89540, #27AE60, #A855F7, #F2C94C, #56CCF2, #FF6FB1
```

### 7. Graphe Multimodal

Le graphe represente le moodboard comme un reseau esthetique.

Noeuds:

- images;
- couleurs;
- objets;
- symboles;
- textures;
- styles;
- emotions;
- affects;
- compositions;
- esthetiques.

Edges:

- `image_similarity`;
- `attention_similarity`;
- `color_affinity`;
- `emotion_affinity`;
- `style_affinity`;
- `texture_affinity`;
- `composition_affinity`;
- `co_occurrence`;
- `aesthetic_match`.

Les noeuds de modalite ne sont pas places arbitrairement. Leur vecteur est le centroide pondere des images auxquelles ils sont associes.

Le layout 2D cherche a approximer les distances latentes:

1. UMAP cosine si disponible;
2. t-SNE cosine;
3. MDS cosine;
4. PCA;
5. cercle de fallback.

Donc plus deux vecteurs sont eloignes dans l'espace latent, plus leurs noeuds doivent etre eloignes visuellement.

## Calcul De Purete

La purete finale n'est pas une seule heuristique. Elle fusionne plusieurs signaux.

### A. Purete Latente

La purete latente mesure la coherence avant l'analyse spectrale.

Elle combine:

- coherence intra-cluster;
- separation inter-cluster;
- dominance du cluster principal;
- concentration des concepts/symboles;
- coherence de style;
- coherence affective;
- coherence emotionnelle;
- coherence couleur;
- couverture des modalites;
- marge entre l'esthetique dominante et la secondaire;
- penalite d'outliers.

Formules principales:

```text
cluster_cohesion = (intra_similarity + 1) / 2
cluster_separation = (intra_similarity - inter_similarity + 1) / 2

latent_cluster_purity =
  0.34 * cluster_cohesion
  + 0.28 * cluster_separation
  + 0.18 * cluster_dominance
```

Puis:

```text
latent_purity =
  (latent_cluster_purity + modality_convergence)
  * outlier_factor
  * (0.82 + aesthetic_margin * 0.18)
```

### B. Analyse Spectrale

Le graphe est transforme en signal esthetique latent.

On construit:

```text
A = matrice d'adjacence ponderee
D = matrice des degres
L = D - A
```

Puis on calcule les valeurs propres:

```text
eigenvalues = eig(L)
```

Interpretation:

- spectre concentre: moodboard harmonique;
- spectre fragmente: plusieurs regimes forts;
- spectre bruite: dissonance ou outliers;
- 2-3 pics: hybride stable.

Metrics produites:

- `spectralGap`;
- `normalizedSpectralGap`;
- `harmonicityScore`;
- `spectralPurityScore`;
- `dissonanceScore`;
- `hybridizationScore`;
- `aestheticRegimeCount`;
- `dominantAestheticFrequency`;
- `distanceToTargetHarmony`.

### C. Purete Finale

La purete finale est la valeur commune affichee dans l'interface, le graphe et le PDF.

```text
final_purity =
  0.56 * latent_purity
  + 0.24 * spectral_purity
  + 0.08 * harmonicity
  + 0.05 * world_mood_confidence
  + 0.04 * aesthetic_margin
  + 0.03 * modality_coverage
```

Puis:

```text
hybridation = 1 - final_purity
```

Cette fusion evite que la purete soit uniquement une mesure de similarite visuelle. Elle tient compte de la structure du graphe, des modalites detectees, de la palette, des clusters et de la proximite avec la memoire esthetique.

## Bento Optimizer

Le Bento optimizer est optionnel et desactive par defaut dans l'interface.

Quand il est active, il ne choisit pas seulement un nombre d'images par page. Il genere plusieurs plans:

- densites differentes;
- ordres d'images differents;
- grilles et templates differents;
- assignations differentes des images aux slots.

Chaque plan est score selon:

- crop/orientation;
- placement des images fortes dans les hero slots;
- harmonie locale des couleurs;
- flow visuel entre images voisines;
- equilibre luminosite/contraste/saturation;
- coherence de regroupement;
- hierarchie de tailles;
- densite cible selon le mode.

Modes:

- `Balanced`: compromis general.
- `Editorial`: moins dense, plus de respiration.
- `Dense`: plus d'images par page.
- `Clustered`: cherche davantage a regrouper les images proches visuellement.

Le code est dans:

```text
src/moodboard/bento/optimizer.py
```

Cette version est deterministe. A terme, elle peut etre remplacee par un Bento Ranker entraine sur les compositions validees par l'utilisateur.

## Export PDF Et PNG

`/api/generate` construit les pages Bento, lance ou reutilise l'analyse, puis serialize:

- PDF seul;
- PNG seul;
- ZIP avec PDF, PNGs et `analysis.json`.

Le PDF ajoute des pages d'analyse:

- resume esthetique;
- scores principaux;
- gradient et palette;
- tags par modalite;
- carte latente en nuage de points;
- analyse spectrale;
- clusters, aesthetics proches et outliers.

## Databases Locales

Les donnees locales sont centralisees dans `database/`.

Refresh complet:

```bash
source .venv/bin/activate
python scraping/refresh_database.py
```

Refresh couleurs:

```bash
python scraping/refresh_database.py --colors
```

Refresh esthetiques:

```bash
python scraping/refresh_database.py --aesthetics --source all --limit 500
```

Sources:

- Aesthetics Wiki via metadata;
- CARI metadata-only, sans redistribution d'assets;
- Name That Color pour enrichir les noms de couleurs.

## World Learning

Le dossier `world_learning/` est reserve a l'apprentissage offline du world model.

Objectif:

1. convertir des datasets heterogenes vers un schema commun;
2. produire des embeddings et observations par modalite;
3. entrainer ou calibrer des artifacts;
4. charger ces artifacts dans l'app au demarrage, sans reentrainer pendant l'utilisation.

Schema cible:

```json
{
  "image_id": "...",
  "embedding": [],
  "palette": [],
  "objects": [],
  "symbols": [],
  "textures": [],
  "style_tags": [],
  "emotion_tags": [],
  "affect_tags": [],
  "composition": [],
  "dataset_origin": "...",
  "aesthetic_score": null,
  "neighbors": []
}
```

Datasets prevus ou adaptes:

- LAION;
- LAION-Aesthetics;
- Behance BAM;
- BAID;
- AVA;
- Polyvore;
- Danbooru;
- dossiers locaux.

Artifacts charges par l'app:

```text
data/trained_models/aesthetic_text_index_v1.json
data/trained_models/fusion_calibrator_v1.json
data/trained_models/world_sample_index_v1.json
data/trained_models/world_mood_classifier_v1.json
```

Important: l'app ne doit pas entrainer les modeles a chaque lancement. Elle charge les artifacts disponibles et degrade proprement s'ils sont absents.

## API Analyse

La structure principale de `/api/analyze` contient:

```text
images[]
globalProfile
palette
scores
clusters
outliers
aestheticMatches
graph
spectralAnalysis
modelStatus
cache
worldModel
```

Les noeuds du graphe ont la forme:

```json
{
  "id": "symbol-halo",
  "type": "symbol",
  "label": "Halo",
  "cluster": 0,
  "clusterColor": "#5D71FC",
  "weight": 0.72,
  "x": 0.12,
  "y": -0.31,
  "associatedImages": ["image-1", "image-4"]
}
```

Les edges ont la forme:

```json
{
  "source": "image-1",
  "target": "symbol-halo",
  "type": "co_occurrence",
  "weight": 0.81
}
```

## Tests Rapides

Compilation des modules principaux:

```bash
.venv/bin/python -m py_compile \
  app/server.py \
  src/moodboard/analysis/moodboard_analyzer.py \
  src/moodboard/analysis/purity.py \
  src/moodboard/analysis/spectral.py \
  src/moodboard/bento/render.py \
  src/moodboard/bento/optimizer.py \
  src/moodboard/bento/pdf_export.py
```

Verifier le serveur:

```bash
curl http://127.0.0.1:8787/api/health
curl http://127.0.0.1:8787/api/models
```

Datasets de test locaux:

```text
dataset-test-1/
dataset-test-2/
dataset-test-3/
```

## Notes D'Architecture

- Le serveur est volontairement simple: `ThreadingHTTPServer` et fichiers statiques.
- Les caches sont en memoire process-local, pas persistants.
- Les modeles sont charges paresseusement et conserves dans `ML_MODEL_CACHE`.
- L'analyse est protegee par un lock car UMAP, Numba et certains runtimes ML ne sont pas toujours safe en concurrence.
- Le graphe est la representation centrale: il relie images, couleurs, symboles, objets, affects et esthetiques.
- La purete est un score produit par le pipeline, pas une valeur UI dupliquee.
- Les bases externes sont metadata-only quand les droits d'assets ne sont pas clairs.

