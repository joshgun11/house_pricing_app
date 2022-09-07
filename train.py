from parser import KParseArgs
from model import KModel
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import r2_score
import pickle
import sys

class Ktrain():

    def __init__(self) -> None:
        pass

    def load_data(self,path):
        df = pd.read_csv(path)
        return df 

    def create_target_feature(self,data,test_size):
        columns = list(data.columns)
        columns.remove('MEDV')
        print(len(columns))
        features = data[columns]
        #print(len(features))
        X_train,X_test,y_train,y_test = train_test_split(features,data['MEDV'],test_size = test_size,random_state=43)
        return X_train,X_test,y_train,y_test

    def train(self,args):
        path = 'model_data.csv'
        df = self.load_data(path)
        X_train,X_test,y_train,y_test = self.create_target_feature(df,args.test_size)
        model_type = args.model_type

        models =  KModel()
        if model_type == 'linear_reg':
            model = models.linear_regression()
        elif model_type =='lasso':
            model = models.lasso_regresion(args.alpha)

        elif model_type =='ridge':
            model = models.ridge_regresion(args.alpha)

        elif model_type =='forest':
            model = models.tree_regresion()
        print('Selected model:' + str(model_type))

        model.fit(X_train,y_train)
        print('Model succesfully trained')
        predictions = model.predict(X_test)
        r_Score = r2_score(y_test, predictions)
        print('Score of the builde model: ',r_Score)
        if args.save:
            pickle.dump(model, open('models/'+str(model_type)+'.pkl', 'wb'))


if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    Ktrain().train(args)
        
        


