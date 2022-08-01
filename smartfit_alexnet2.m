% clc
clear all
addpath(genpath(pwd));
folderpath='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\';
rootFolder = strcat(folderpath, '\im_Alexnet_database');
imds = imageDatastore(rootFolder, 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

[trainingSet, validationSet] = splitEachLabel(imds, 0.9, 'randomize');
%%
net =  alexnet;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(trainingSet.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])


layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

miniBatchSize = 10;
valFrequency = floor(numel(trainingSet.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(trainingSet,lgraph,options);






% 
% 
% 
% layersTransfer = net.Layers(1:end-3);
% numClasses = numel(categories(trainingSet.Labels))
% 
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];
% 
% % pixelRange = [-30 30];
% % imageAugmenter = imageDataAugmenter( ...
% %     'RandXReflection',true, ...
% %     'RandXTranslation',pixelRange, ...
% %     'RandYTranslation',pixelRange);
% % augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
% %     'DataAugmentation',imageAugmenter);
% % 
% % augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);
% % 
% % 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',9, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',validationSet, ...
%     'ValidationFrequency',3, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% netTransfer = trainNetwork(trainingSet,layers,options);
% % netTransfer = net;

[YPred,scores] = classify(net,validationSet);


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