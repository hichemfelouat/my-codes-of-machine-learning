from sklearn.feature_extraction.text import CountVectorizer
texts = [
    "blue car and blue window",
    "black crow in the window",
    "i see my reflection in the window"
]
vec = CountVectorizer(binary=True)
vec.fit(texts)
print([w for w in sorted(vec.vocabulary_.keys())])
X = vec.transform(texts).toarray()
print(X)

import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))


vec = CountVectorizer(binary= False)
vec.fit(texts)
print([w for w in sorted(vec.vocabulary_.keys())])
X = vec.transform(texts).toarray()
print(X)

import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))


from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
vec.fit(texts)
print([w for w in sorted(vec.vocabulary_.keys())])
X = vec.transform(texts).toarray()
import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))
