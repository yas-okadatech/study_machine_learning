import os
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)

# print(vectorizer)

content = ['How to format my hard disk', ' Hard disk format problems ']
x = vectorizer.fit_transform(content)
print( vectorizer.get_feature_names() )
print( x.toarray().transpose()) #　表3-1と同じ。[文章1の出現回数, 文章2の出現回数]

DIR = './data/ch3'
posts = [open (os.path.join(DIR,f)).read() for f in sorted(os.listdir(DIR))]
X_train = 