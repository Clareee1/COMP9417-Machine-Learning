import numpy as np
import matplotlib.pyplot as plt
from preproc import preproc
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def sum_stats(features, feature_names, labels, label_names, highest = 10, tfidf = False):
    #Takes features, feature names, labels and label names and prints
    #summary statistics to the terminal.
    
    #For data that has been normalised by term frequency inverse document frequency
    if tfidf:
        
        #Group articles by topic
        array_features = features.toarray()
        no_words = array_features.shape[1]
        max_word = np.zeros((len(label_names), no_words))
        for i in range(len(labels)):
            for j in range(no_words):
                
                max_word[labels[i]][j] = max(max_word[labels[i]][j], array_features[i][j])
        
        
        #Print highest tf-idf score words for each topic in corpus
        for i in range(len(label_names)):
    
            freq_words = sorted(zip(max_word[i], feature_names), reverse = True)[:highest]
            print('========================')
            print('The ', highest, ' words with the highest TF-IDF score across ', label_names[i], ' articles are:', sep='')
            
            for j in range(len(freq_words)):
                print(freq_words[j][1], ' ' * (20 - len((freq_words[j][1]))),
                      '{:.4f}'.format(freq_words[j][0]), sep='')
        
        
    #For data that hasn't been normalised by term frequency inverse document frequency
    else:
        #Group articles by topic
        array_features = features.toarray()
        no_words = array_features.shape[1]
        total = np.zeros((len(label_names), no_words))
        for i in range(len(labels)):
            total[labels[i]] = total[labels[i]] + array_features[i]
        
        
        #Print summary statistics for corpus...
        print('There are ', '{:,}'.format(int(np.sum(total))), ' words in the corpus.', sep='')
        print('There are ', '{:,}'.format(len(feature_names)), ' unique words in the corpus.', sep='')
        
        
        #...then foreach topic in corpus
        for i in range(len(label_names)):
            
            no_words = np.sum(total[i])
            unique_words = np.count_nonzero(total[i])
            freq_words = sorted(zip(total[i], feature_names), reverse = True)[:highest]
            
            #Summary statistics for each topic...
            print('========================')
            print('The total number of words in', label_names[i], 'articles is: ', '{:,}'.format(int(no_words)))
            print('The total number of unique words in', label_names[i], 'articles is: ', '{:,}'.format(int(unique_words)))
            print('The ', highest, ' most frequent words are:', sep='')
            
            
            #... and most frequent words
            for j in range(len(freq_words)):
                
                print(freq_words[j][1], ' ' * (20 - len((freq_words[j][1]))), 
                      '{:,}'.format(int(freq_words[j][0])), 
                      ' ' * (8 - len('{:,}'.format(int(freq_words[j][0])))), 
                      '(', '{:.2f}'.format(freq_words[j][0] / no_words * 100),'%)', sep='')
    
    return



def article_freq(labels, label_names, colours = ["#C00000", "#FF0000", "#FFC000", "#FFFF00", "#92D050", 
           "#00B050", "#00B0F0", "#0070C0", "#7030A0", "#000000", "#BFBFBF"]):
    #Takes labels and label names and outputs the distribution of articles
    #by topic in both print and graph form.
    
    
    article_count = []
    total = len(labels)
    print('========================')
    print('Article Distribution (%)')
    print('========================')
    
    
    #Count number of occurrences for each topic
    for i in range(len(label_names)):
    
        n = np.count_nonzero(labels == i)
        freq = n / total
        article_count.append(freq)
        print(label_names[i], ' ' * (40 - len(label_names[i])), '{:.2f}'.format(freq * 100), '%', sep = '')
    
    
    #Plot article distribution
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(label_names, article_count, color = colours)
    plt.xticks(rotation = 90)
    plt.show()
    
    return



def t_SNE(features, labels, label_names, no_comps = 100, no_iters = 5,
          colours = ["#C00000", "#FF0000", "#FFC000", "#FFFF00", "#92D050", 
           "#00B050", "#00B0F0", "#0070C0", "#7030A0", "#000000", "#BFBFBF"],
          max_features = None, max_range = 1):
    #Dimensionality reduction
    #See below site for more details
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    svd = TruncatedSVD(n_components = no_comps, n_iter = no_iters, random_state = 1)
    svd_fitted = svd.fit_transform(features)
    
    #t-SNE decomposition
    #See below site for more details. Note it is highly recommended to reduce the dimensionality of the data
    #if it is sparse (i.e. lots of zeroes in the array like in bag of words)
    #https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    embedded_features = TSNE(n_components = 2).fit_transform(svd_fitted)
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in np.unique(labels):
        ix = np.where(labels == i)
        ax.scatter(embedded_features[:, 0][ix], embedded_features[:, 1][ix], 
                   c = colours[i], label = label_names[i], s = 50)
    ax.legend()
    title = 'T-SNE using max_features = ' + str(max_features) + ' and max_range = ' + str(max_range)
    plt.title(title)
    plt.show()
    return



TRAINCSV = 'training.csv'
TESTCSV = 'test.csv'
colours = ["#C00000", "#FF0000", "#FFC000", "#FFFF00", "#92D050", 
           "#00B050", "#00B0F0", "#0070C0", "#7030A0", "#000000", "#BFBFBF"]
run_stats = 0
test_tsne_inputs = 0
test_tfidf_inputs = 1
range_n_comps = [100, 500, 750, 1000, 1250, 1500, 2000]
range_n_iters = [5, 10, 15, 20]
range_max_features = [None, 3000, 2500, 1500, 1000, 500]
range_max_range = [1, 2, 3]

if test_tfidf_inputs == 1:
    for i in range(len(range_max_features)):
        for j in range(len(range_max_range)):
            
            #Preprocess features and labels 
            f = preproc(TRAINCSV, tfidf = True, max_range = range_max_range[j], max_features = range_max_features[i])
            
            #Store output into variables
            train_features = f.features[0]
            feature_names = f.features[1]
            train_labels = f.labels[0]
            list_of_labels = f.labels[1]
            
            #Run t-SNE
            t_SNE(train_features, train_labels, list_of_labels, 100, 5, colours, range_max_features[i], range_max_range[j])
            print('Finished T-SNE for pair:')
            print('Max features:', range_max_features[i])
            print('Max range:', range_max_range[j])
else:
    #Preprocess features and labels 
    f = preproc(TRAINCSV, tfidf = True)
    
    #Store output into variables
    train_features = f.features[0]
    feature_names = f.features[1]
    train_labels = f.labels[0]
    list_of_labels = f.labels[1]
    
    
    #Run summary statistics
    if run_stats == 1:
        sum_stats(train_features, feature_names, train_labels, list_of_labels, tfidf = True)
        sum_stats(train_features, feature_names, train_labels, list_of_labels, tfidf = False)
        article_freq(train_labels, list_of_labels)
    
    
    #T-SNE loop
    if test_tsne_inputs == 1:
        for i in range(len(range_n_comps)):
            for j in range(len(range_n_iters)):
                t_SNE(train_features, train_labels, list_of_labels, range_n_comps[i], range_n_iters[j], colours)
                print('Finished T-SNE for pair:')
                print('N-comps:', range_n_comps[i])
                print('N-iters:', range_n_iters[j])
    #t_SNE(train_features, train_labels, list_of_labels, 500, 7, colours)
            
