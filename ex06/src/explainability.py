# Created by Edoardo Mosca
# Edited by Melf Harders and Stefan Wege
# For Social Computing/Social Gaming

import numpy as np
import shap

# Computes the prediction and shap values for a given tweet, then prints out various data about the prediction.
# Returns a force plot for each label, that can be displayed.
def shap_explain_text(input_sentence, label, model_to_explain, label_mapping, explainer, words_indexing, use_matplotlib=False):
    
    # Computes the models prediction
    prediction = model_to_explain.predict(np.array([input_sentence,]))[0]
    
    # Computes shap values for our input sentence
    shap_values = explainer.shap_values(np.array([input_sentence, ]))

    # Decides the predicted class by selecting the highest score from the 3-dimensional prediction vector that our model returns
    predicted_class = np.argmax(prediction)
    
    # Assign the real class from the One Hot Encoded Waseem Hovy labels
    real_class = np.argmax(label)
    
    # Prints out various data about our prediction
    print('PREDICTION')
    for index, labelname in enumerate(label_mapping):
        print('{} : {}'.format(labelname, prediction[index]))
        
    print('\nREAL CLASS: {}'.format(label_mapping[real_class]))

    # Transform the indices to words
    words = words_indexing
    num2word = {}
    for w in words.keys():
        num2word[words[w]] = w
    tweet = list(map(lambda x: num2word.get(x, "NONE"), input_sentence))
    
    # Prints the original (padded) tweet
    print('\nTWEET')
    print(rebuild_sentence(tweet))
    print('\n')
	
    feature_names = np.array(tweet)
	
    # Creates a force plot for each label and returns a list of all plots
    plots = []
	
    for i in range(len(label_mapping)):
        plots.append(shap.force_plot(explainer.expected_value[i], shap_values[i][0], feature_names, matplotlib=use_matplotlib))
		
    return plots


# Rebuilds string from list
def rebuild_sentence(split_sentence):
    filtered_sentence = list(filter((lambda x: x != '<pad>'), split_sentence))
    return " ".join(filtered_sentence)