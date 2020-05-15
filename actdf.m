function res = actdf(sfvalue)
% derivative of sigmoid activation function
% sfvalue - value of sigmoid activation function (!)

	res = 1/1*(1-sfvalue.*sfvalue);
