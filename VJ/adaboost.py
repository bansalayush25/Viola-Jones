import numpy as np
from VJ.haarfeatures import HaarFeature
from VJ.haarfeatures import FeatureTypes

def learn(pos_img, neg_img, min_feat_height=1, max_feat_height=-1, min_feat_width=1, max_feat_width=-1, num_classifier =2):
    pos_len = len(pos_img)
    neg_len = len(neg_img)
    len_img = pos_len + neg_len
    img_height, img_width = pos_img[0].shape
    
    max_feat_height = img_height if max_feat_height==-1 else max_feat_height
    max_feat_width = img_width if max_feat_width==-1 else max_feat_width
    
    pos_weights = np.ones(pos_len) * 1.0 / 2*pos_len
    neg_weights = np.ones(neg_len) * 1.0 / 2*neg_len
    
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(pos_len), np.ones(neg_len) * -1.0))
    
    images = np.zeros((pos_faces.shape[0] + neg_faces.shape[0], pos_faces.shape[1], pos_faces.shape[2]))
    images[:pos_faces.shape[0]] = pos_faces
    images[pos_faces.shape[0]:] = neg_faces
    
    features = create_features(img_height, img_width, min_feat_height, max_feat_height, min_feat_width, max_feat_width)
    len_features = len(features)
    feature_index = list(range(len_features))
    
    num_classifier = len_features if num_classifier==-1 else num_classifier
    
    votes = np.zeros((len_img, len_features))
    for i in range(len_img):
        for j in range(len_features):
            votes[i][j] = get_feature_vote(feature[j], images[i])
  
    classifiers = []
            
    for i in range(num_classifier):
        
        classification_errors = np.zeros(len(feature_index))        
        weights *=1.0/np.sum(weights)
        
        
        for f in range(len(feature_index)):            
            f_idx = feature_index[f]
            error = sum(map(lambda img_idx : weights[img_idx] if labels[img_idx]!=votes[img_idx, f_idx] else 0, range(len_img)))
            
            classification_errors[f] = error
        
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]

        print best_feature_idx
        
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight

        classifiers.append(best_feature)

        weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(len_img))))

        feature_index.remove(best_feature_idx)
            
    return classifiers

def create_features(img_height, img_width, min_feat_height, max_feat_height, min_feat_width, max_feat_width):
    features = []
    for feature in FeatureTypes:
        feature_start_width = max(feature[0], min_feat_width)
        for feature_width in range(feature_start_width, max_feat_width, feature[0]):
            feature_start_height = max(feature[1], min_feat_height)
            for feature_height in range(feature_start_height, max_feat_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarFeature(feature, (x,y), feature_width, feature_height, 0, 1))
                        features.append(HaarFeature(feature, (x,y), feature_width, feature_height, 0, -1))
    return features

def get_feature_vote(feature, img):
    return feature.get_vote(img)