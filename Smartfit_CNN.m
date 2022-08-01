clc
clear all
addpath('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\');

imds = imageDatastore('simulated_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomize');
%%
net = alexnet;
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingSet.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);


options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(netTransfer,augimdsValidation);
idx = randperm(numel(validationSet.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation)
plotconfusion(YValidation,YPred)

%%

%%
% I=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% featureVector = encode(bag,I);
% bar(featureVector)
%%
img = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)