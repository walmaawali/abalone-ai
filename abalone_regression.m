%% Run if you have regression output (like age)

%% MLP Code
% First run the data_processing code
data_preprocessing

% initialize an MLP
hidden_layers = [2 2];
net = feedforwardnet(hidden_layers);

% configure the MLP
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.3;
net.divideParam.testRatio = 0.0000;
net.performFcn = "mse";

% train the MLP
net = train(net, trainX, trainY);

% testing the MLP on test set
y = net(testX);

% compute the performance metrics
mse = perform(net,y,testY);

disp("Testing MSE of the MLP: "+ mse);
disp("Testing RMSE of the MLP: "+ sqrt(mse));


%% ANFIS Code

% initialize ANFIS with 2 membership functions, and 100 epochs
[anfis err] = ANFIS(trainX, trainY, 2, 100);

% test the anfis
y = evalfis(anfis,testX');

% plot the error
figure;
hold on;
xlabel('Epochs')
ylabel('Root Mean Squared Error')
title("Error Curve - ANFIS")
hold on

plot(err,"LineWidth",2);
legend("Age")

% compute testing performance metrics
mse = immse(testY',y);

disp("Testing MSE of the ANFIS: "+ mse);
disp("Testing RMSE of the ANFIS: "+ sqrt(mse));


% Plot the resultant FIS for ANFIS
figure;
plotmf(anfis,'input',1)
title("Input 1 - Membership functions")

figure
plotmf(anfis,'input',2)
title("Input 2 - Membership functions")

% ..repeat if you have more

%% _______________________DON'T CHANGE! ___________________________________

function [out_fis error] = ANFIS(input, target, num_membership, num_epochs)
    foptions = genfisOptions('GridPartition');
    foptions.NumMembershipFunctions = num_membership;
    foptions.InputMembershipFunctionType = "gbellmf";
    x = input'; y = target';
    in_fis  = genfis(x,y,foptions);
    options = anfisOptions;
    options.InitialFIS = in_fis;
    options.EpochNumber = num_epochs;
    [out_fis error] = anfis([x,y],options);
end
