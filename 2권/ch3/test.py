import sys
sys.path.append('..')
import common.util
from common.util import create_contexts_target, preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
window_size = 1
contexts, target = create_contexts_target(corpus, window_size)
print(contexts)
print(target)
print(corpus)