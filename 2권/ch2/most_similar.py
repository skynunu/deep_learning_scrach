import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

print('You say goodbye and I say hello에서 유사도 top5')
most_similar('you', word_to_id, id_to_word, C, top=5)