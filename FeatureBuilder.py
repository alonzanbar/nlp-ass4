

from parse_tree_utils import get_parse_tree_path, extract_dep_map, build_graph


def extract_features(pair, doc):
    en1, en2 = pair
    start = en1.end + 1
    end = en2.start
    prev_word = doc[en1.start - 1].text if en1.start > 0 else 'None'
    next_word = doc[en2.end].text if en2.end < len(doc) else 'None'
    graph = build_graph(doc)
    dep_map, typed_dep_map = extract_dep_map(en1,en2,doc, graph)
    words_set = set([t.text for t in doc[start:end]])
    consitutient_path = "-".join(get_parse_tree_path(doc.text, en1.root.text, en2.root.text))
    features = {

        'entity1-head': en1.root,
        'entity2-type': en2.label,
        'entity2-head': en2.root,
        'concatenatedtypes': en1.label_ + en2.label_,
        'syntactic-path': "-".join([w.pos_ for w in doc[start:end]]),
        'base-syntactic-path': "-".join([w.tag_ for w in doc[start:end]]),
        'word-before-entity1': prev_word,
        'word-after-entity2': next_word,
        'dep-path': "-".join(typed_dep_map),
        'dis_ent_distance': len(typed_dep_map) / 3,
        'between-entities-word': words_set,
        'consitutient_path': consitutient_path

    }

    return features
