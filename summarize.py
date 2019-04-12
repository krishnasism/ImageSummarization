def summarize(image):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    import os 
    
    from PIL import Image
        
    words=[]
    sentences=image.keys()
    for sentence in sentences:
        words.extend(word_tokenize(sentence))

    #sentences = sent_tokenize(text)
    #tweets=[]
    #for sentence in sentences:
    #    tweets.extend(sentence.split('b\''))
    #sentences=tweets
    #print(sentences)

    
    
    stop_words = set(stopwords.words("english"))
    f=open(os.path.dirname(os.path.realpath(__file__))+"/stopwords.txt")
    for stops in f.read().split():
        stop_words.add(stops)
    #print(sentences)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    
    
    
    c1 = list()
    c2 = list()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        #print ("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            if(i == 0):
                c1.append(terms[ind])
            else:
                c2.append(terms[ind])
    
    #print("\n")
    #print("\n")
    #print("\n")
    
    #print("Cluster 1 :")
    #print(c1)
    #print("\n")
    #print("Cluster 2 : ")
    #print(c2)
    
    sentence_score={}
    sc = 1.0
    for sentence in sentences:
        sc=1.0
        for word in c1:
            #print("\n* "+ word)
            if word in sentence.lower():
                if sc<=0:
                    sc=0
                if sentence in sentence_score.keys():
                    sentence_score[sentence]+=sc
                    sc = sc-0.05
                    #print(sentence_score[sentence])
                else:
                    sentence_score[sentence]=sc
                    sc = sc-0.05
                    #print(sentence_score[sentence])
                    
    for sentence in sentences:
        sc=1.0
        for word in c2:
            #print("\n* "+ word)
            if word in sentence.lower():
                if sc<=0:
                    sc=0
                if sentence in sentence_score.keys():
                    sentence_score[sentence]+=sc
                    sc = sc-0.05
                    #print(sentence_score[sentence])
                else:
                    sentence_score[sentence]=sc
                    sc = sc-0.05
                    #print(sentence_score[sentence])
    
    
        
    greatest=0
    
    highest = ""
    
    for sentence in sentence_score.keys():
        print(str(sentence_score[sentence]) + "is the score of" + sentence )
        if sentence_score[sentence]>greatest:
            greatest=sentence_score[sentence]
            highest=sentence
    

    print("Most important image = " + highest)
    print("Most important image = " + image[highest])
    img=Image.open(image[highest])
    img.show()
            