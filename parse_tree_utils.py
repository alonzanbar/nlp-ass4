import nltk
from nltk.tag.stanford import CoreNLPNERTagger


def get_parse_tree_path(sent_str,word1,word2):
    tagger = CoreNLPNERTagger(url='http://localhost:9000')
    (parsed), = tagger.parse(sent_str.split())
    #parsed.pretty_print()
    pt = nltk.tree.ParentedTree.convert(parsed)
    leaf1 = find_leaf(pt, word1)
    leaf2 = find_leaf(pt,word2)
    if leaf1 is None or  leaf2 is None:
        return []
    path1,path2 = find_path(leaf1,leaf2)
    dir_path1 = intersperse([pt[a].label() for a in path1], '-u-')
    dir_path2 = intersperse([pt[a].label() for a in path2],'-d-')
    dir_path2.insert(0,'-d-')
    return dir_path1+dir_path2

def find_path(leaf1,leaf2):
    parent1 = leaf1
    parent2 = leaf2
    path1=[]
    path2=[]
    while parent1:
        path1.append(parent1.treeposition())
        parent1 = parent1.parent()
    p2id = {k:v for v,k in enumerate(path1)}
    while parent2:
        p2position = parent2.treeposition()
        if parent2.treeposition() in p2id:
           return path1[1:p2id[p2position]+1],list(reversed(path2[1:]))
        path2.append(parent2.treeposition())
        parent2 = parent2.parent()

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def find_leaf(pt, word):
    st = []
    st.append(pt[0])
    while len(st) > 0:
        c = st.pop()
        if isinstance(c[0], unicode):
            if c[0] == word:
                #print (word, c.label())
                return c
        else:
            for child in c:
                st.append(child)


if __name__=="__main__":
    sentences = 0
    sent_str = "An enraged Nikita Khrushchev instructed Soviet ships"# to ignore President Kennedy 's naval blockade during the Cuban missile crisis , but the order was reversed just hours before an inevitable confrontation , according to a new book ."
    ents  = getparsetree(sent_str,'Nikita','Soviet')
    print ents
    pass
