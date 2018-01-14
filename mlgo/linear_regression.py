# TODO


class LinearRegressionUni:
    """
    Linear Regression for univariable problems.
    fits training data and predicts test data given.
    """

    def __init__(self):
        self.theta0 = 10
        self.theta1 = 5
        self.trainX = list()
        self.trainY = list()


    def __get_total_cost(self):
        """
        calculate current total cost for current value of thetas
        :return:
        """
        cost_function = 0
        total_data = len(self.trainX)
        for feature,actual_target_class in zip(self.trainX, self.trainY):
            hypothesis = self.theta0+self.theta1*feature
            cost_function += (( hypothesis - actual_target_class)**2)/(2*total_data)
        return cost_function


    def __minimize_cost(self,learning_rate):
        """
        minimizes cost as per current values of theta
        :return:
        """
        num_of_training_data = len(self.trainX)
        derivative_wrt_theta0_sum = 0
        derivative_wrt_theta1_sum = 0
        for feature,actual_target_class in zip(self.trainX, self.trainY):
            hypothesis = self.theta0 + (self.theta1 * feature)
            derivative_wrt_theta0_sum += (hypothesis - actual_target_class)
            derivative_wrt_theta1_sum += (hypothesis - actual_target_class) * feature

        tmp0 = self.theta0 - learning_rate * (derivative_wrt_theta0_sum)/num_of_training_data
        tmp1 = self.theta1 - learning_rate * (derivative_wrt_theta1_sum)/num_of_training_data

        self.theta0 = tmp0
        self.theta1 = tmp1

    def train(self,trainX,trainY,epoch,learning_rate):
        """
        trains the model according to given training data
        :param trainX:
        :param trainY:
        :return:
        """
        if type(trainX) != type([]) or type(trainY) != type([]):
            print("please pass both as list")
            return False

        self.trainX = trainX
        self.trainY = trainY

        for i in range(epoch):
            total_cost = self.__get_total_cost()
            # if i%10 ==0:
            print("iteration no:{}".format(i))
            print("weights: "+str(self.theta0)+" "+str(self.theta1))
            print("cost: {}".format(total_cost))
            self.__minimize_cost(learning_rate)


    def predict(self,testX):
        """
        predicts the target class for given list of features
        :param testX:
        :return:
        """
        prediction = []
        for feature in testX:
            prediction.append(self.theta0 + self.theta1 * feature)
        return prediction

class LinearRegressionMulti:
    """
       Linear Regression for multivariable problems.
       fits training data and predicts test data given.
       """

    def __init__(self):
        self.thetas = []
        self.trainX = list()
        self.trainY = list()

    def product_list(self,lst1,lst2):
        """
        returns products element wise
        :param lst1:
        :param lst2:
        :return:
        """
        fsum = 0
        for x1,x2  in zip(lst1,lst2):
            fsum += x1 * x2
        return fsum


    def __get_total_cost(self):
        """
        calculate current total cost for current value of thetas
        :return:
        """
        cost_function = 0
        total_data = len(self.trainX)
        for feature, actual_target_class in zip(self.trainX, self.trainY):
            hypothesis = self.product_list(self.thetas,feature)
            cost_function += ((hypothesis - actual_target_class) ** 2) / (2 * total_data)
        return cost_function


    def __minimize_cost(self, learning_rate):
        """
        minimizes cost as per current values of theta
        :return:
        """
        num_of_training_data = len(self.trainX)
        derivatives = [0]*len(self.trainX[0])
        for feature, actual_target_class in zip(self.trainX, self.trainY):
            hypothesis = self.product_list(self.thetas,feature)
            for indx,x in enumerate(feature):
                derivatives[indx] +=(hypothesis-actual_target_class) * x

        temp_thetas = self.thetas
        final_thetas = []
        for indx,theta in enumerate(temp_thetas):
            new_theta = theta - learning_rate * derivatives[indx] / num_of_training_data
            final_thetas.append(new_theta)

        self.thetas = final_thetas


    def __add_bias(self,trainX):
        """
        adds X0 or bias to the feature list provided
        :param trainX:
        :return:
        """
        final_lst = []
        for row in trainX:
            temp_list =[1]
            temp_list.extend(row)
            final_lst.append(temp_list)
        return final_lst


    def train(self, trainX, trainY, epoch, learning_rate):
        """
        trains the model according to given training data
        :param trainX:
        :param trainY:
        :return:
        """
        if type(trainX) != type([[]]) or type(trainY) != type([]):
            print("please pass both as list")
            return False

        self.trainX = self.__add_bias(trainX)
        self.trainY = trainY
        self.thetas = [ randint(1,10) for x in self.trainX[0]]

        for i in range(epoch):
            total_cost = self.__get_total_cost()
            # if i%10 ==0:
            print("iteration no:{}".format(i))
            print("weights: ")
            print(self.thetas)
            print("cost: {}".format(total_cost))
            self.__minimize_cost(learning_rate)

    def predict(self, testX):
        """
        predicts the target class for given list of features
        :param testX:
        :return:
        """
        prediction = []
        for feature in testX:
            added_bias = [1]
            added_bias.extend(feature)
            prediction.append(self.product_list(self.thetas,added_bias))
        return prediction



if __name__ == '__main__':
    from random import randint, random

    lr = LinearRegressionMulti()
    trainX = [[randint(0, 10),randint(0,10)] for i in range(100)]
    trainY = []
    for row in trainX:
        y  =  5000+( 800 * row[0])+(500* row[1])
        trainY.append(y)

    lr.train(trainX,trainY,100000,0.0005)
    print  lr.predict([[0,1],[2,3],[4,5]])