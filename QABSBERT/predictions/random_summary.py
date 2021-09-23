import random

src_docs = open('documents.txt', 'r').readlines()
tgt_summaries = open('targets.txt', 'r').readlines()
qabsbert_candidates = open('qabsbert_only_170k.txt', 'r').readlines()
qabsbert_200k_candidates = open('qabsbert_only_200k.txt', 'r').readlines()
rm_weighting_candidates = open('qabsbert_w_rm_weighting.txt', 'r').readlines()
rm_term_candidates = open('qabsbert_w_rm_term.txt', 'r').readlines()

num_docs = len(src_docs)

index = random.randint(0,num_docs)

print("Sorted source document:")
src_doc = src_docs[index].replace('[CLS]','').replace('[SEP]','').replace(' ##','').replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', '').replace(' [unused2] ', '').replace('[unused2]', '').strip()
print(src_doc)
print()

print("Target summary:")
print(tgt_summaries[index].replace('<q>', '. '))

print("QABSBERT 200k summary:")
print(qabsbert_200k_candidates[index].replace('<q>', '. '))

print("QABSBERT 170k summary:")
print(qabsbert_candidates[index].replace('<q>', '. '))

print("QABSBERT 160k + 10k with RM loss weighting:")
print(rm_weighting_candidates[index].replace('<q>', '. '))

print("QABSBERT 160k + 10k with RM loss term:")
print(rm_term_candidates[index].replace('<q>', '. '))