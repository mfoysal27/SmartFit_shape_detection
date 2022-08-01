% clc
clear all
addpath(genpath(pwd));
folderpath='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\';
rootFolder = 'im_Alexnet_database\';
imds = imageDatastore(rootFolder, 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

[trainingSet, validationSet] = splitEachLabel(imds, 0.9, 'randomize');
%%
net =  alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingSet.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
%     'DataAugmentation',imageAugmenter);
% 
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);
% 
% 
options = trainingOptions('sgdm', ...
    'MiniBatchSize',9, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(trainingSet,layers,options);
% netTransfer = net;

[YPred,scores] = classify(netTransfer,validationSet);


YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation)
plotconfusion(YValidation,YPred)
%%
% confMatrix = evaluate(categoryClassifier, trainingSet);
%%

% %%
% img = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% [labelIdx, scores] = predict(categoryClassifier, img);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)