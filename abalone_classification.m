%% Run if you have classification output (like Gender)

%% MLP Code

% First run the data_processing code
data_preprocessing

% initialize an MLP
hidden_layers = [5 5 5];
net = feedforwardnet(hidden_layers);

% configure the MLP
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.3;
net.divideParam.testRatio = 0.0000;
net.performFcn = "mse";

% Sigmoid activation for the output layer
net.layers{length(hidden_layers)}.transferFcn = 'logsig'; 

% train the MLP
net = train(net, trainX, trainY);

% testing the MLP on test set
y = net(testX);

% convert to binary output by making the maximum equals to 1
% and others 0
out = convert_output(y');

% compute the performance metrics
mse = perform(net,y,testY);
accuracy = compute_accuracy(out, testY');

disp("Testing accuracy of the MLP: "+ accuracy*100+"%");
disp("Testing MSE of the MLP: "+ mse);
disp("Testing RMSE of the MLP: "+ sqrt(mse));

% if using gender as output
plot_confusion(testY',out)

%% ANFIS Code

% initialize and train 3 anfis (for each output)
% each with 2 membership functions, and 100 epochs
[anfis1 err1] = ANFIS(trainX, trainY(1,:), 2, 100);
[anfis2 err2] = ANFIS(trainX, trainY(2,:), 2, 100);
[anfis3 err3] = ANFIS(trainX, trainY(3,:), 2, 100);

% test the anfis
out1 = evalfis(anfis1,testX');
out2 = evalfis(anfis2,testX');
out3 = evalfis(anfis3,testX');

% combine the outputs into one output
system_output = [out1 out2 out3];
out = convert_output(system_output);

% plot the error
err = [err1 err2 err3];
figure;
hold on;
xlabel('Epochs')
ylabel('Root Mean Squared Error')
title("Error Curve - ANFIS")
hold on

for i=1:3
    plot(err(:,i),"LineWidth",2)
end
legend("Female", "Infant", "Male")

% compute testing performance metrics
accuracy = compute_accuracy(out, testY');
mse = immse(testY',system_output);

disp("Testing accuracy of the ANFIS: "+ accuracy*100+"%");
disp("Testing MSE of the ANFIS: "+ mse);
disp("Testing RMSE of the ANFIS: "+ sqrt(mse));

plot_confusion(testY',out)

% Plot the resultant FIS for ANFIS 1
figure;
plotmf(anfis1,'input',1)
title("Input 1 - Membership functions")

figure
plotmf(anfis1,'input',2)
title("Input 2 - Membership functions")

%% _______________________________________________________________

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



%%
function out = convert_output(y)
    out = zeros([length(y), 3]);
    for i=1:length(y)
        [m, idx] = max(y(i,:));
        out(i, idx) = 1;
    end
end

%%
function accuracy = compute_accuracy(y_predict, y_true)
    count = 0;
    for i=1:length(y_true)
        Y1 = y_predict(i,:);
        Y2 = y_true(i,:);
        count = count + (sum(xor(Y1,Y2)) < 1);
    end
    accuracy = (count / length(y_true));
end

%%
function [arr] = convert_labels(y)

    arr = zeros([length(y),1]);
    for i=1:length(y)
        [m idx] = max(y(i,:));
        arr(i)=idx;
    end
end

%%
function [] = plot_confusion(y1, y2)
    L1 = convert_labels(y1);
    L2 = convert_labels(y2);
    C = confusionmat(L1,L2);
    figure;
    confusionchart(C,{'Female', 'Infant','Male'});
end