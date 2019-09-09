from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)

# print(vectorizer)

content = ['How to format my hard disk', ' Hard disk format problems ']
x = vectorizer.fit_transform(content)
print( vectorizer.get_feature_names() )
