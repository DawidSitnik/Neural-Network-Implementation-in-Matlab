function [hidlw outlw terr] = backprop(tset, tslb, inihidlw, inioutlw, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

% 1. Set output matrices to initial values
	hidlw = inihidlw;
	outlw = inioutlw;
	
% 2. Set total error to 0
	terr = 0;
	
% foreach sample in the training set
	for i=1:rows(tset)

		% 3. Set desired output of the ANN
		desired_output = -1*ones(1, size(outlw,2));
    	desired_output(tslb(i)) = 1;

		% 4. Propagate input forward through the ANN
    	values_hidden = actf([tset(i, :) 1] * hidlw);
    	values_output = actf([values_hidden 1] * outlw);
		
		% 5. Adjust total error
		terr += sum((values_output - desired_output).^2);
		
		% 6. Compute delta error of the output layer
		delta_out = (values_output - desired_output) .* actdf(values_output);
		
		% 7. Compute delta error of the hidden layer
   	 	delta_hid = delta_out * transpose(outlw(1:end-1,:)) .* actdf(values_hidden);
		
		% 8. Update output layer weights
    	outlw -= lr * transpose([values_hidden 1]) * delta_out;
		
		% 9. Update hidden layer weights
    	hidlw -= lr * transpose([tset(i, :) 1]) * delta_hid;
	end
