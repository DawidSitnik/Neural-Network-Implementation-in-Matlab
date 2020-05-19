# rand seed was always set to this value while experiments
rand("state", "reset");
rand("seed", 1);

[tvec tlab tstv tstl] = readSets();

tlab = tlab+1;
tstl = tstl+1;

[hlnn olnn] = crann(columns(tvec), 200, 10);

n_epochs = 150;
errors_in_epochs_training = [];
errors_in_epochs_testing = [];

desired_error= 0.1258;

for epoch = 1:n_epochs

  [hlnn olnn terr] = backprop_modified(tvec, tlab, hlnn, olnn, 0.001);
  
  % error callculation for training dataset
  clsRes = anncls(tvec, hlnn, olnn);
  error = sum(clsRes!=tlab)/size(tlab,1);
  errors_in_epochs_training(end+1) = error;

  % error callculation for testing dataset
  clsRes_test = anncls(tstv, hlnn, olnn);
  error_test = sum(clsRes_test!=tstl)/size(tstl,1);
  errors_in_epochs_testing(end+1) = error_test;

  % stop after getting satisfying error
  if error_test <= desired_error
    break
  end
end
clc
plot(errors_in_epochs_training)
hold on
plot(errors_in_epochs_testing)
title('Training and Testing Error During Backprop - Modified Version')
legend('training','testing')
hold off
 
%file_name = strcat('./workspaces/workspace_modified', datestr(date));
%save(file_name);

display("result for training data:")
clsRes_train = anncls(tvec, hlnn, olnn);
cfmx_train = confMx(tlab, clsRes_train);
errcf_train = compErrors(cfmx_train)

display("result for testing data:")
clsRes_test = anncls(tstv, hlnn, olnn);
cfmx_test = confMx(tstl, clsRes_test);
errcf_test = compErrors(cfmx_test)