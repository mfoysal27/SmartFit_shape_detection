
clear all
addpath('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\');


imds = imageDatastore('simulated_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');
% imds = imageDatastore('smartfit_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');


[trainingSet, validationSet] = splitEachLabel(imds, .7, 'randomize');
%%
tic
%%
% I=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% featureVector = encode(bag,validationSet);
% bar(featureVector)
% scatter(featureVector, 1:500)

% %%
% validationSet.Files
% validationimg = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\imdatabase\rectangle\18_.png');
% 
% [labelIdx, scores] = predict(categoryClassifier, validationSet);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)

bag = bagOfFeatures(trainingSet, 'StrongestFeatures', 0.99);

%%

imageIndex = invertedImageIndex(bag);

addImages(imageIndex, trainingSet);
queryImage = imread('simulated_database\hourglass\1 (4).jpg');
queryROI = [4 4 36 36]; 

figure
imshow(queryImage)
rectangle('Position',queryROI,'EdgeColor','yellow')

%%

 img = imread('simulated_database\hourglass\1 (4).jpg');
 points = detectSURFFeatures(rgb2gray(img));
[featureVector words] = encode(bag, img);

[B,I] = maxk(featureVector, 3);


wordlocation=unique(words.Location);

% [B1 F1]=mode(words.WordIndex);
% location= wordlocation(B1);
% B2=mode
% B3=

% a = unique(featureVector);
% out = [a,histcounts(featureVector(:),a)];

% N = histcounts(featureVector, 500);

featureVector_3=zeros( 1, 500);
words_3=zeros( 1, 500);
featureVector_3(I)=featureVector(I);
words_3(I)=words.WordIndex(I);
wordssss=unique(words.WordIndex);


% Plot the histogram of visual word occurrences
figure
bar(featureVector_3, 'BarWidth', 10)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')


%%
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
toc
predict(categoryClassifier, trainingSet);
confMatrix = evaluate(categoryClassifier, trainingSet)
accuracy=mean(diag(confMatrix))

% img=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\simulated_database\hourglass\download(1).png');
% evaluate(categoryClassifier, img)

predict(categoryClassifier, validationSet);
confMatrix = evaluate(categoryClassifier, validationSet)
accuracy=mean(diag(confMatrix))
%%
% I=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% featureVector = encode(bag,validationSet);
% bar(featureVector)
% scatter(featureVector, 1:500)

% %%
% validationSet.Files
% validationimg = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\imdatabase\rectangle\18_.png');
% 
% [labelIdx, scores] = predict(categoryClassifier, validationSet);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)
% 
% [labelIdx, scores] = predict(categoryClassifier, trainingSet);
% categoryClassifier.Labels(labelIdx)
% confMatrix = evaluate(categoryClassifier, validationSet);
%%
% img = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% [labelIdx, scores] = predict(categoryClassifier, img);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)