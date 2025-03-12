Un **Large Language Model (LLM)** est un modèle d'intelligence artificielle entraîné sur de grandes quantités de texte pour générer, comprendre et manipuler le langage naturel. Voici les étapes principales de son fonctionnement :

---

## 1. Prétraitement des Données

Avant d'entraîner un LLM, il faut préparer les données.

### 1.1 Tokenisation

Le texte est découpé en unités appelées **tokens** (mots, sous-mots ou caractères).

```
Phrase : "Les LLMs sont puissants."
Tokens : ["Les", "LLMs", "sont", "puissants", "."]
```

### 1.2 Normalisation

- Conversion en **minuscules** : "Chat" → "chat"
- Suppression des **ponctuations** : "bonjour!" → "bonjour"
- Suppression des **espaces multiples** : "bonjour toi" → "bonjour toi"

```python
import re
text = re.sub(r'\s+', ' ', text)  # Remplace les espaces multiples par un seul
text = re.sub(r'[^\w\s]', '', text)  # Supprime la ponctuation
```

---

## 2. Vectorisation des Tokens

Les tokens sont convertis en vecteurs numériques.

### 2.1 One-Hot Encoding

Chaque mot est représenté par un vecteur où un seul élément est activé :

$$\text{mot}_i \rightarrow \mathbf{v}_i \in \mathbb{R}^N, \quad \text{où } N \text{ est la taille du vocabulaire}$$

Exemple pour un vocabulaire de 5 mots :

```
"chat" -> [1, 0, 0, 0, 0]
"chien" -> [0, 1, 0, 0, 0]
```

### 2.2 Word Embeddings (Word2Vec, Glove, etc.)

Une approche plus efficace est d'utiliser des vecteurs denses :

$$\mathbf{v}_i = \begin{bmatrix} 0.2 \\ -0.3 \\ 0.5 \\ \vdots \\ 0.1 \end{bmatrix} \in \mathbb{R}^d$$

Ces vecteurs capturent la sémantique :

```
cos( "chat", "chien" ) ≈ 0.85 (proches)
cos( "chat", "voiture" ) ≈ 0.1 (éloignés)
```

---

## 3. Modèle de Réseau Neuronal

Les LLMs utilisent des **transformers**, un type de réseau neuronal introduit par Vaswani et al. (2017).

### 3.1 Architecture Transformer

```
+--------------------+
|    Input Tokens   |
+--------------------+
          |
          v
+--------------------+
|  Embedding Layer  |
+--------------------+
          |
          v
+--------------------+
| Multi-Head Self-  |
|   Attention       |
+--------------------+
          |
          v
+--------------------+
|    Feed Forward   |
+--------------------+
          |
          v
+--------------------+
|     Output Layer  |
+--------------------+
```

### 3.2 Mécanisme d'Attention

L'attention permet au modèle de se concentrer sur les parties importantes d'une phrase.

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

#### Formule du Score d'Attention

où :
- $Q$ = Matrice des **requêtes** (queries)
- $K$ = Matrice des **clés** (keys)
- $V$ = Matrice des **valeurs** (values)
- $d_k$ = Dimension des clés

---

## 4. Entraînement

Un LLM est entraîné en minimisant une **fonction de perte**, comme l'entropie croisée :

$$L = - \sum_{i} p_i \log(\hat{p}_i)$$

où :

- $p_i$ = vraie probabilité du mot $i$
- $\hat{p}_i$ = probabilité prédite par le modèle

L'optimisation est réalisée avec **Adam** (descente de gradient améliorée).

---

## 5. Génération de Texte

Une fois entraîné, un LLM génère du texte en **prédictant le mot suivant**.

### Exemples d'approches :

- **Greedy Decoding** : Choix du mot avec la probabilité la plus haute.
- **Beam Search** : Explore plusieurs possibilités avant de choisir.
- **Sampling** : Ajoute de la diversité en échantillonnant selon la distribution de probabilité.


Exemple de génération avec un LLM :

```
Input: "Il était une fois"
Output: "un roi qui régnait sur un vaste empire."
```

---

## Conclusion

Un LLM repose sur plusieurs étapes clés :

1. **Prétraitement** (tokenisation, nettoyage)
2. **Vectorisation** (one-hot, embeddings)
3. **Réseau de neurones** (transformers, attention)
4. **Entraînement** (minimisation de la perte)
5. **Génération de texte** (prédiction du mot suivant)

Cette architecture puissante permet aux LLMs d'être utilisés dans diverses applications comme la traduction, les chatbots et l'analyse de texte.
