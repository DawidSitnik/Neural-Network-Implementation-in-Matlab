function [hidlw outlw terr] = backprop_modified(tset, tslb, inihidlw, inioutlw, lr, b_size, inertia, set_point)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN
  if nargin<6
    b_size = 1;
  end
  if nargin<7
    inertia=0;
  end
  if nargin<8
    set_point=1;
  end
% 1. Set output matrices to initial values
	hidlw = inihidlw;
	outlw = inioutlw;
	
% 2. Set total error to 0
	terr = 0;
	cnt = 0
% foreach sample in the training set
  d_hidden_prev = 0;
  d_output_prev = 0;
	for i=1:b_size:rows(tset)-b_size
    if cnt == 10000
      i
      cnt = 0 ;
    end
    cnt+=1;
		% 3. Set desired output of the ANN
		desired_output = -set_point*ones(b_size, size(outlw,2));
    for label=1:b_size
      desired_output(label, tslb(i+label-1)) = set_point;
    end
		% 4. Propagate input forward through the ANN
		% remember to extend input [tset(i, :) 1]
    values_hidden = actf([tset(i:i+b_size-1, :) ones(b_size,1)] * hidlw);
    values_output = actf([values_hidden ones(b_size,1)] * outlw);
		% 5. Adjust total error (just to know this value)
		terr += sum(sum((values_output - desired_output).^2));
		% 6. Compute delta error of the output layer
		% how many delta errors should be computed here?
		delta_out = (values_output - desired_output) .* actdf(values_output);
		% 7. Compute delta error of the hidden layer
		% how many delta errors should be computed here?
    delta_hid = delta_out * outlw(1:end-1,:)' .* actdf(values_hidden);
		delta_out;
    delta_hid;
		% 8. Update output layer weights
		d_hidden = lr * [values_hidden ones(b_size,1)]' * delta_out/b_size;
    outlw -= d_hidden + inertia*d_hidden_prev;
    d_hidden_prev = d_hidden;
		% 9. Update hidden layer weights
    d_output = lr * [tset(i:i+b_size-1, :) ones(b_size,1)]' * delta_hid/b_size;
    hidlw -= d_output + inertia*d_output_prev;
    d_output_prev = d_output;
    end
	end

