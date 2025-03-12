import re
import numpy as np

def clean_text(text):
    """
    Nettoies le texte donné en enlevant la ponctuation, les espaces inutiles + diviser le texte en unités plus petites

    s+ -> un ou plusieurs caractères d'espacement
    w -> caractère alphanumérique (lettres et chiffres)
    """

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remplacer les espaces multiples par un seul
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation

    return text

def tokenize(text):
    return text.split()

def vocab_creation(token):
    """
    Construit un vocabulaire sous forme {"mot": index, "mot": 2}
    """

    vocab = {word: idx for idx, word in enumerate(set(token))}

    return vocab

def context_pairs(tokens, window_size=1):
    """
    Génère des fenêtres contextuelles -> tuple contenant les mots avant de window_size et après

    "Ceci est un exemple" -> [('ceci', 'est'), ('est', 'ceci'), ('est', 'un'), ('un', 'est'), ('un', 'exemple'), ('exemple', 'un')]
    """

    pairs = []

    for i, word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1) # Prévient index négatif ou index trop grand

        context = [tokens[j] for j in range(start, end) if j != i]
        for ctx_word in context:
            pairs.append((word, ctx_word))
    return pairs

def compute_loss(target, context, target_vec, context_vec):
    score = np.dot(target_vec, context_vec)  # Produit scalaire
    loss = -np.log(sigmoid(score))  # Entropie croisée
    return loss

# Fonction de rétropropagation et mise à jour des vecteurs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update_vectors(target, context, target_vec, context_vec, learning_rate):
    score = np.dot(target_vec, context_vec)
    grad_target = sigmoid(score) - 1
    grad_context = sigmoid(score) - 1
    # Mise à jour des poids
    target_vec -= learning_rate * grad_target * context_vec
    context_vec -= learning_rate * grad_context * target_vec
    return target_vec, context_vec

def train(pairs, vocab):

    inv_vocab = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    embedding_dim = 5  # Dimension de l'embedding
    learning_rate = 0.01

    # Initialisation aléatoire des vecteurs (poids)
    target_vectors = np.random.randn(vocab_size, embedding_dim)
    context_vectors = np.random.randn(vocab_size, embedding_dim)

    for target_word, context_word in pairs:
        target_idx = vocab[target_word]
        context_idx = vocab[context_word]
    
        # Récupérer les vecteurs pour le mot cible et le mot de contexte
        target_vec = target_vectors[target_idx]
        context_vec = context_vectors[context_idx]
    
        # Calculer la perte et mettre à jour les vecteurs
        loss = compute_loss(target_word, context_word, target_vec, context_vec)
        target_vectors[target_idx], context_vectors[context_idx] = update_vectors(target_word, context_word, target_vec, context_vec, learning_rate)

        # Affichage contextualisé avec les mots et la perte au format lisible
        print(f"({target_word} -> {context_word}) Loss: {loss:.4f}")

    return (target_vectors, context_vectors)


text = "Ceci est un exemple"
text = clean_text(text)
print(text)
    
token = tokenize(text)
vocab = (vocab_creation(token))
print(vocab)
pairs = context_pairs(token)
print(pairs)

print(train(pairs, vocab))