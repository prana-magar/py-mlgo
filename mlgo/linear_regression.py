

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


if __name__ == '__main__':
    from random import randint
    lr = LinearRegressionUni()
    trainX = [randint(0, 10) for i in range(100)]
    trainY = []
    for val in trainX:
        y  =  5000+( 800 * val)
        trainY.append(y)

    lr.train(trainX,trainY,100000,0.0005)
    print  lr.predict([0,1,2,3,4,5])