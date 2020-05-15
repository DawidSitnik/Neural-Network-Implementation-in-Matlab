# rand seed was always set to this value while experiments
rand("state", "reset");
rand("seed", 0);

[tvec tlab tstv tstl] = readSets();

load tiny.txt
tlab = tiny(:,1);
tvec = tiny(:,2:end);

[hlnn olnn] = crann(columns(tvec), 100, 3);
n_epochs = 150;
errors_in_epochs = []
positive_classifications = []
error = 0.9
desired_error= 0

for epoch = 1:n_epochs

  % setting learning rate basing on error
  lr = 0.0005 + (0.01-0.0005)*(error-0.1)/(0.9-0.1);
  [hlnn olnn terr] = backprop_modified(tvec, tlab, hlnn, olnn, lr, 1, 0.5, 1);
  
  clsRes = anncls(tvec, hlnn, olnn);
  error = sum(clsRes!=tlab)/size(tlab,1);

  errors_in_epochs(end+1) = error;

  % stop after getting satisfying error
  if error <= desired_error
    break
  end
end

plot(errors_in_epochs)
 
file_name = strcat('./workspaces/workspace_', datestr(date))
save(file_name)

display("result for training data:")
clsRes_train = anncls(tvec, hlnn, olnn);
cfmx_train = confMx(tlab, clsRes_train);
errcf_train = compErrors(cfmx_train)

display("result for testing data:")
clsRes_test = anncls(tstv, hlnn, olnn);
cfmx_test = confMx(tstl, clsRes_test);
errcf_test = compErrors(cfmx_test)