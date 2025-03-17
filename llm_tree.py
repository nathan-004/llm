import random
import re
import os

class InitialNode():
    def __init__(self):
        self.childs = {}

    def print(self):
        for child in self.childs:
            self.childs[child].print()

    def add(self, phrase):
        if phrase[0] in self.childs:
            self.childs[phrase[0]].add(phrase)
        else:
            self.childs[phrase[0]] = Node(phrase, 0)

    def generate(self, phrase=None):
        if phrase is None:
            self.choose = [key for key in self.childs for _ in range(self.childs[key].n)]
            value = random.choice(self.choose)

            return value + self.childs[value].generate()


class Node():
    """
    phrase:liste de mots
    """
    def __init__(self, phrase, depth=0):
        self.value = phrase[0]
        self.childs = {}
        self.depth = depth
        self.n = 1
        if phrase[1:] != []:
            if phrase[1] in self.childs:
                self.childs[phrase[1]].add(phrase[1:])
            else:
                self.childs[phrase[1]] = Node(phrase[1:], self.depth+1)

    def print(self):
        print("   " * self.depth, self.value)
        for child in self.childs:
            self.childs[child].print()

    def add(self, phrase):
        self.n += 1

        if phrase[1:] != []:
            if phrase[1] in self.childs:
                self.childs[phrase[1]].add(phrase[1:])
            else:
                self.childs[phrase[1]] = Node(phrase[1:], self.depth+1)

    def generate(self, phrase=""):
        if self.childs == {}:
            return phrase

        self.choose = [key for key in self.childs for _ in range(self.childs[key].n)]
        value = random.choice(self.choose)

        return self.childs[value].generate(phrase+" "+value)



def clean_text(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text)  # Remplace les espaces multiples par un seul

    corpus = []
    for phrase in text.split("."):
        corpus.append(re.sub(r'[^\w\s]', '', phrase).split(" "))  # Supprime la ponctuationphrase.split(" "))
        while "" in corpus[-1]:
            corpus[-1].pop(corpus[-1].index(""))

    print(corpus)

    return corpus

root = InitialNode()

def train(corpus):
    for phrase in corpus:
        if phrase == []:
            continue
        root.add(phrase)
    root.print()

def generate(text=None):
    if text is None:
        text = ""
        print(root.generate())
    else:
        text.split(" ")

def get_texts():
    # Retourne le corpus contenant tous les textes dans le dossier Texte
    corpus = []

    for filename in os.listdir("./Textes"):
        with open("./Textes/"+filename, "r", encoding="UTF-8") as f:
            corpus.extend(clean_text(f.read()))

    return corpus



if __name__ == "__main__":

    corpus = get_texts()

    train(corpus)

    for _ in range(50):
        generate()
