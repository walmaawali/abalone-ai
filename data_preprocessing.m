clear; clc;
% load the dataset
data = readtable("abalone.data","FileType","text");

% select input feature
features = ["gender", "length", "diameter", "height", "total_weight", "flesh_weight", "gut_weight", "shell_weight", "age"];
data.Properties.VariableNames = features;

% split into training and testing (you can change the ratio and input data)
input_features = ["length", "diameter", "height", "total_weight"];
outputs = ["gender"];
train_ratio = 0.80;     % the remaining will be used for testing

[trainX, trainY, testX, testY] = data_split(data, input_features, outputs, train_ratio);

%%
function [trainX, trainY, testX, testY] = data_split(data, input_features, outputs, train_ratio)
    Y = [];
    X = [];
    X_gend=[];
    Y_gend=[];
    delete_i = 0;

    for i = 1:length(input_features)
        feature = input_features(i);
        if feature == "gender"
            X_gend = encode_gender(data);
            delete_i = i;
        else
            X_temp = data{:, feature};
            X(:,i) = X_temp;
        end
    end

    if ~isempty(X_gend)
        col = size(X, 2);
        X(:,col+1:col+3) = X_gend;
        if col > 0
            X(:,delete_i) = [];
        end
    end

    for j = 1:length(outputs)
        output = outputs(j);
        if output == "gender"
            Y_gend = encode_gender(data);
            delete_i = j;
        else
            Y_temp = data{:, output};
            Y(:,j) = Y_temp;
        end
    end

    if ~isempty(Y_gend)
        col = size(Y, 2);
        Y(:,col+1:col+3) = Y_gend;
        if col > 0
            Y(:,delete_i) = [];
        end
    end
    
    rr = randperm(height(data));
    
    trainIdx = floor(height(data)*train_ratio);
    
    trainX = X(rr(1:trainIdx), :)';
    trainY = Y(rr(1:trainIdx), :)';
    
    testX = X(rr(1+trainIdx:end), :)';
    testY = Y(rr(1+trainIdx:end), :)';
end

%%
function [res] = encode_gender(data)
    labels = cellstr(data.gender);
    labels = categorical(labels);
    %categories(labels)

    % result will be in this order: F, I, M
    res = onehotencode(labels, 2);
end