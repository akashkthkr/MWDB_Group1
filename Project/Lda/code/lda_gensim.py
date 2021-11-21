import gensim
import numpy as np
import dataset

feature_type = 'hog'
num_topics = 10


def get_type_from_filename(filename):
    return filename.split('-')[1]


def create_corpus(object_feature_mapping):
    corpus = []
    for o in object_feature_mapping:
        row = [(index, freq) for index, freq in enumerate(o)]
        corpus.append(row)
    return corpus


def create_id2word(feature_type, feature_length):
    id2w = dict()
    for i in range(feature_length):
        id2w[i] = feature_type + str(i)
    return id2w


json = dataset.open_json('all_database.json')
image_ids = json.keys()
print(image_ids)

print(len(image_ids))
n_components = [1, 5, 20, len(image_ids) - 1]

object_feature_mapping = np.array([json[id]['features'][feature_type] for id in image_ids])
print("object_feature_mapping=", object_feature_mapping, ' shape=', object_feature_mapping.shape)

corpus = create_corpus(object_feature_mapping)
num_features = len(object_feature_mapping[0])
id2word = create_id2word(feature_type, num_features)

lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=num_topics,
                                   random_state=100,
                                   chunksize=100,
                                   passes=10,
                                   per_word_topics=True)

# print(lda_model[create_corpus([object_feature_mapping[0]])[0]])
topics = lda_model.get_topics()
print(topics, topics.shape)

inference = lda_model.get_document_topics(create_corpus([object_feature_mapping[0]]))
print(inference[0])

print(lda_model.print_topics(num_topics=num_topics))

print(lda_model.top_topics())
