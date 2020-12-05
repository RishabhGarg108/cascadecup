def SplitLabels(data, label):
    '''
    Split the dataset into inputs and labels.
    
    @param data The pandas dataframe which we need to split.
    @param label String value which is the name of output label.
    
    @return X Inputs
    @return y Outputs
    '''
    
    X = data.drop(label, axis=1)
    y = data[label]
    return X,y

def DropFeatures(features, data):
    '''
    Drops the list of features from data.
    
    @param features List of features to drop.
    @param data Dataframe from which we want to drop features.
    
    @return dropped_data The dataset with dropped features.
    '''
    
    dropped_data = data.drop(features, axis = 1)
    return dropped_data
    
def Score(score_func, y_true, y_pred, **args):
    '''
    Calculates the score of predictions.
    
    @param score_func The function which we need as a measure of score.
    @param y_true True labels.
    @param y_pred Predicted labels.
    @param args Additional arguments fed to the score_func.
    
    @return calc_score The score value calculated by the score_func.
    '''
    
    calc_score = score_func(y_true, y_pred, **args)
    return calc_score
       
  
