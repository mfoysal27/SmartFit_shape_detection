clc
clear all
addpath(genpath(pwd))


imds = imageDatastore('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\smartfit_database_preprocessed', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

[trainingSet, validationSet] = splitEachLabel(imds, 0.9, 'randomize');
%%
finalFeature=zeros(length(trainingSet.Files), 2,  64);
scatterplot=zeros(length(trainingSet.Files), 2);
for i = 1 : length(trainingSet.Files)

       I1 = (imread(trainingSet.Files{i}));
if length(size(I1))==3
    I1=rgb2gray(I1);
end
       points1 = detectSURFFeatures(I1);
       [features2, valid_points1] = extractFeatures(I1, points1);
%        figure; imshow(I1); hold on;
%        plot(valid_points1.selectStrongest(40),'showOrientation',true);
%        columnvector = zeros(1,64); 
%        finalFeature(i, 1, :) = features2(1, :);
%        finalFeature(i, 2, :) = features2(2, :);
%        scatter(finalFeature(i, 1, 5), finalFeature(i, 2, 5))
       scatterplot(i, 1)=features2(1, 5);
       scatterplot(i, 2)=features2(2, 5);
hold on;
%        sprintf('%s %d','finalFeature = ' , finalFeature)
    end
% finalFeature;
scatter (scatterplot(:, 1), scatterplot(:, 2))
%%
species=trainingSet.Labels;
X = scatterplot(:,1:2);
y = categorical(species);
labels = categories(y);
gscatter(X(:,1),X(:,2),species,'', 'ox^s');
% xlabel('Sepal length');
% ylabel('Sepal width');

classifier = fitcknn(X,y);
% classifier = fitcecoc(X,y);
x1range = min(X(:,1)):.0001:max(X(:,1));
x2range = min(X(:,2)):.0001:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
   predictedspecies = predict(classifier,XGrid);
   
   hold on
   

     gscatter(xx1(:), xx2(:), predictedspecies,'cmky');
xlabel('Feature Dimension');
ylabel('Feature Dimension');
title('k Nearest Neighbor')


legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal')
hold on
% gscatter(X(:,1),X(:,2),species,'g', 'o');
species=trainingSet.Labels;
X = scatterplot(:,1:2);
y = categorical(species);
labels = categories(y);
gscatter(X(:,1),X(:,2),species,'', 'ox^s');