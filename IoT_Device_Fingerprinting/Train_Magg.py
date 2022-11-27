import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    final = pd.read_csv('data/Traces_Magg/M_agg_features.csv')
    X = final.drop(['Device_ID'],axis=1)
    Y = final.Device_ID

    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    print('Training Random Forest .....')
    rf = RandomForestClassifier()
    rf.fit(X_train,Y_train)
    Y_RF_pred=rf.predict(X_test)

    filename = 'Models/M_agg_fingerprinting.sav'
    pickle.dump(rf, open(filename, 'wb'))
    
    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))


    print('Training Complete. Testing Model: ')

    print(classification_report(Y_test,Y_RF_pred,zero_division = 0))

if __name__ == '__main__':
    main()