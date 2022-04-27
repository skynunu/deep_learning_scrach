
text = "You say goodbye and i say hello"
text = text.lower()
text = text.replace('.',' .')
print("text 전처리 : ", text)
words = text.split(' ')
print("word로 분리 : ", words)

word_to_id = {}
id_to_word ={}

for word in words :
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] =word

print("id_to_word : ", id_to_word)
print("word_to_id : ", word_to_id)