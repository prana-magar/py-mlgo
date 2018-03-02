"""
logistic regression in python.
1) for binary classification
2) multiclass classification
"""
import math
import numpy as np
from random import randint, random


class Logistic_Reg_Binary():
    """
    Logistic regression for two class classification.
    """

    def __init__(self):
        self.thetas = []
        self.trainX = []
        self.trainY = []

    def __sigmoid(self,val):
        """
        returns the sigmoid function value for given x.
        1/(1+e^-x)
        :param val:
        :return:
        """
        return 1/(1+math.exp(-val))


    def __get_hypothesis(self,feature_vec):
        """
        returns the value of the hypothesis for given  feature vector.
        :param theta_vec:
        :param feature_vec:
        :return:
        """
        theta_transpose = np.transpose(self.thetas)
        product = np.dot(theta_transpose,feature_vec)
        # print product
        return self.__sigmoid(product)


    def __get_partial_dev_sum(self,indx):
        """
        returns the sum of partial derivative of cost function wrt theta at given indx
        :param trainX:
        :param indx:
        :return:
        """
        dervative_val = 0
        for feature_row,target_class in zip(self.trainX,self.trainY):
            hypothesis = self.__get_hypothesis(feature_row)
            part_derivative = (hypothesis - target_class) * feature_row[indx]
            dervative_val+=part_derivative
        return dervative_val

    def __minimize_cost(self,learning_rate):
        """
        minimizes the cost for given training data and sets the thetas value for optimal for minimim cost.
        :param learning_rate: rate for the step of derivative.
        :return:
        """
        updated_thetas_vec = []
        number_of_training_data  =len(self.trainX)
        for indx,theta in enumerate(self.thetas):
            partial_derivative_wrt_theta = (self.__get_partial_dev_sum(indx))/number_of_training_data
            new_theta = theta - learning_rate * partial_derivative_wrt_theta
            updated_thetas_vec.append(new_theta)
        self.thetas = updated_thetas_vec

    def __add_bias(self, trainX):
        """
        adds X0 or bias to the feature list provided
        :param trainX:
        :return:
        """
        final_lst = []
        for row in trainX:
            temp_list = [1]
            temp_list.extend(row)
            final_lst.append(temp_list)
        return final_lst


    def __get_cost(self):
        """
        calculates cost for current theta values.
        :return:
        """
        number_of_training_data = len(self.trainX)
        total_cost = 0
        for feature,target_class in zip(self.trainX,self.trainY):
            hypothesis = self.__get_hypothesis(feature)
            cost =  target_class * math.log(hypothesis) + (1 - target_class) * np.log(1 - hypothesis)
            total_cost += cost
        return  -(total_cost/number_of_training_data)

    def normalize_feature(self,features_lst):
        """
        normalizes all features from trainX by
            Fnew = (F - mean)/standard_deviation
        :return:
        """
        new_features = []
        means = np.mean(features_lst,axis=1)
        stds = np.std(features_lst,axis=1)
        for feature_row in self.trainX:
            new_row = []
            for indx,feature in enumerate(feature_row):
                new_feature = (feature - means[indx])/stds[indx]
                new_row.append(new_feature)
            new_features.append(new_row)
        return  new_features


    def train(self,trainX,trainY,epochs,learning_rate):
        """
        trains the model by reducing the cost/error for the hypothesis.
        :param train_X:
        :return:
        """
        self.trainX = trainX
        self.trainX = self.__add_bias(self.trainX)
        # self.trainX = self.normalize_feature(self.trainX)
        print self.trainX
        self.trainY = trainY
        self.thetas = [ randint(1,2) for x in self.trainX[0]]

        for epc in range(0,epochs):
            self.__minimize_cost(learning_rate)
            if epc%10 == 0:
                print(self.thetas)
                print("cost")
                print self.__get_cost()
        print("model trained")


    def predict(self,feature_lst):
        """
        predicts target class for the given feature vector.
        :param feature:
        :return:
        """
        prediction = []
        feature_lst = self.__add_bias(feature_lst)
        # feature_lst = self.normalize_feature(feature_lst)
        for feature_row in feature_lst:
            prediction.append(self.__get_hypothesis(feature_row))
        return prediction



if __name__ == '__main__':
    logisR = Logistic_Reg_Binary()

    trainX = []
    trainY = []
    trainX_class1 = [[0 , 0] for i in range(100)]
    trainY_class1 = [0  for row in trainX_class1]
    trainX_class2 = [[0, 1] for i in range(100)]
    trainY_class2 = [1 for row in trainX_class1]
    trainX_class3 = [[1, 0] for i in range(100)]
    trainY_class3 = [1 for row in trainX_class1]
    trainX_class4 = [[1, 1] for i in range(100)]
    trainY_class4 = [1 for row in trainX_class1]


    trainX = trainX_class1+ trainX_class2 + trainX_class3 + trainX_class4
    trainY = trainY_class1+ trainY_class2 +trainY_class3 + trainY_class4
    print trainX
    print trainY
    logisR.train(trainX,trainY,1000,0.9)
    print logisR.thetas
    print logisR.predict([[0,0]])
    print logisR.predict([[0,1]])
    print logisR.predict([[1, 0]])
    print logisR.predict([[1, 1]])