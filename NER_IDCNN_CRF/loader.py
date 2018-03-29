import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_entity_tags(path):
    """
    Load supported entity tags from path
    :param path:
    :return: Entity tags list
    """
    tags = []
    count = 0
    for line in codecs.open(path, 'r', 'utf8'):
        count += 1
        item = line.rstrip()
        if item:
            tags.append(item)
    print("total tags: {} from :{}".format(len(tags), path))
    return tags


def load_sentences_by_tag(folder, tags, lower, zeros, max_num):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    :param folder: The folder path of the sentences corpus
    :param tags: Supported tags list
    :param max_num: The maximum sentence number for each tag
    :return: Sentences list
    """
    tags_files = [x + ".txt" for x in tags]

    print(tags_files)

    corpus_files = [os.path.abspath(f)
                    for f in os.listdir(folder)
                    if os.path.isfile(f) and f in tags_files]
    print(corpus_files)

    tag_sentences = [load_sentences(f, lower, zeros, max_num)
                     for f in corpus_files]

    corpus = [item for sublist in tag_sentences for item in sublist]

    print("total corpus: {} from folder: {}".format(len(corpus), folder))

    return corpus

def load_sentences(path, lower, zeros, max_sentence):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    sentence_num = 0
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                    sentence_num += 1
                    if sentence_num % 10000 == 0:
                        print("sentence_num: {}".format(sentence_num))
                    if sentence_num > max_sentence:
                        break
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            assert len(word) >= 2, print("error num: {}".format(num))
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    print("sentences: {} from: {}".format(len(sentences), path))
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [tag_to_id[w[-1]] if w[-1] in tag_to_id else none_index for w in s]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

