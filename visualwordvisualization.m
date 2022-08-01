
clear all
addpath('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\');


imds = imageDatastore('simulated_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

[trainingSet, validationSet] = splitEachLabel(imds, .7, 'randomize');

bag = bagOfFeatures(trainingSet, 'StrongestFeatures', 0.99);

%%

% imageIndex = invertedImageIndex(bag);
% 
% addImages(imageIndex, trainingSet);
% queryImage = imread('simulated_database\hourglass\1 (4).jpg');
% queryROI = [80 78 20 20]; 
% 
% figure
% imshow(queryImage)
% rectangle('Position',queryROI,'EdgeColor','yellow')

%%

 img = imread('simulation_image.jpg');
%  img_data = imageDatastore('simulated_database\hourglass\1 (4).jpg');

%  bag = bagOfFeatures(img_data);
%  points = detectSURFFeatures(rgb2gray(img));
[featureVector words] = encode(bag, img);

[B,I] = maxk(featureVector, 500);
% I=I(2:end);
featureVector_2=zeros( 1, 500);
featureVector_2=featureVector;
featureVector_2(I(1))=0;

figure
imshow((img))
[zz z]=maxk(featureVector_2, 5);
featureVector_3=zeros( 1, 500);
featureVector_3(I(2:4))=featureVector(I(2:4));


% for l=z
for l=I(2:4)
indexword=words.WordIndex;
z1=min(find(indexword==(l)));
queryROI1 = [words.Location(z1, :)-8 20 20]
rectangle('Position',queryROI1,'EdgeColor','yellow');
hold on
end
%%
% Plot the histogram of visual word occurrences
figure
bar(featureVector_2, 'BarWidth', 10, 'FaceColor', 'red')
hold on
bar(featureVector_3, 'BarWidth', 10, 'FaceColor', 'blue')

title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence');


%%
% wordlocation=unique(words.Location);
% 
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

