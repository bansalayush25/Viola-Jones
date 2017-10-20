'''FACE DETECTION USING VIOLA JONES ALGORITHM

It basically is a 4 step process.

1. INTEGRAL IMAGE 
2. HAAR FEATURES
3. ADABOOST CLASSIFIER
4. CASCADING WEAK CLassifiers

'''

import VJ.cascade
import VJ.adaboost as ab
import VJ.integralimage as ii
import VJ.utils

if __name__=="__main__":
    
    pos_train_path = 'faces/train/pos/*.png'
    neg_train_path = 'faces/train/neg/*.png'
    
    pos_train_faces = utils.load(pos_train_path)
    neg_train_faces = utils.load(neg_train_path)
    
    pos_faces = np.array(list(map(ii.convert_integral, pos_train_faces)))
    neg_faces = np.array(list(map(ii.convert_integral, neg_train_faces)))

    print pos_faces.shape, neg_faces.shape
    
    classifiers = ab.learn(pos_faces, neg_faces, num_classifier=20, min_feat_height=4, max_feat_height=10, min_feat_width=4, max_feat_width=10)
    
    pos_test_path = 'faces/test/pos/*.png'
    neg_test_path = 'faces/test/neg/*.png'
    
    pos_test_faces = utils.load(pos_test_path)
    neg_test_faces = utils.load(neg_test_path)
    
    pos_test_faces = np.array(list(map(ii.convert_integral, pos_test_faces)))
    neg_test_faces = np.array(list(map(ii.convert_integral, neg_test_faces)))
    
    print pos_test_faces.shape, neg_test_faces.shape
    
    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(utils.ensemble_all(pos_test, classifiers))
    correct_non_faces = len(neg_test) - sum(utils.ensemble_all(neg_test, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(pos_test_faces))
      + '  (' + str((float(correct_faces) / len(pos_test_faces)) * 100) + '%)\n  non-Faces: '
      + str(correct_non_faces) + '/' + str(len(neg_test_faces)) + '  ('
      + str((float(correct_non_faces) / len(neg_test_faces)) * 100) + '%)')
    
main()