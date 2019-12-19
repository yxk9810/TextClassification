#coding:utf-8
import re
import string
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√','《',
          '》','；','，','。','。','、','？'

          ]

def clean_text(x):
    x = str(x)
    for punct in puncts + list(string.punctuation):
        if punct in x:
            x = x.replace(punct, ' ')
    x = re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", x)

    return x

def clean_numbers(x):
    x= re.sub('(([a-zA-Z]{1,})+([0-9]{1,}[a-zA-Z]{0,})){1,}','digit',x)
    x = re.sub('(([0-9]{1,})+([a-zA-Z]{1,})){1,}','digit',x)
    x= re.sub('\d+', 'digit', x)

    return x

def clean_date(x):
    match = re.search(r"(\d{4}([\.\-/|年]{1})){1,}(\d{1,2}([\.\-/|月]{1})){0,}(\d{1,2}(日|)){0,}",x)
    if match:
        x = x.replace(match.group(),' date ')
    else:
        match = re.search(r"(\d{1,2}月){1,}(\d{1,2}(日|)){0,}", x)
        if match:
            x = x.replace(match.group(),' date ')
    return x

from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

def bestThresshold(y_train, train_preds):
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.961, 0.01)):
        tmp[1] = precision_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2] and tmp[1]>=0.8:
            print(classification_report(y_train, np.array(train_preds) > tmp[0], digits=4))
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta, tmp[2]



