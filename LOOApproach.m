clc
clear all
addpath('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\');
imds = imageDatastore('simulated_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');

accuracy=zeros(80, 1);
    imds1=imds.Files;

for i=1:80

    A=imds1;
    A{i}={};
    A= A(~cellfun(@isempty,A))
    imds_LOOCV_train=imageDatastore(A, 'LabelSource', 'foldernames');
    imds_LOOCV_test=imageDatastore(imds1{i}, 'LabelSource', 'foldernames');
    %%
% I=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\test_shape\test1.jpg');
% featureVector = encode(bag,validationSet);
%        points1 = detectSURFFeatures(I1);
%        [features2, valid_points1] = extractFeatures(I1, points1);
%        figure; imshow(I1); hold on;
%        plot(valid_points1.selectStrongest(40),'showOrientation',true);
%        columnvector = zeros(1,64); 
%        finalFeature(i, 1, :) = features2(1, :);
%        finalFeature(i, 2, :) = features2(2, :);
%        scatter(finalFeature(i, 1, 5), finalFeature(i, 2, 5))

% bar(featureVector)
% scatter(featureVector, 1:500)

bag = bagOfFeatures(imds_LOOCV_train, 'StrongestFeatures', 0.99);
%%
% validationSet.Files
validationimg = imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\imdatabase\rectangle\18_.png');
% 
% [labelIdx, scores] = predict(categoryClassifier, validationSet);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx
%%
categoryClassifier = trainImageCategoryClassifier(imds_LOOCV_train, bag);
% toc
img = readimage(imds_LOOCV_test,1);
[labelIdx, score] = predict(categoryClassifier,imds_LOOCV_test);
lbl_predict=string(categoryClassifier.Labels(labelIdx))
lbl_original=string(cellstr(imds_LOOCV_test.Labels))
% test_label=predict(categoryClassifier, imds_LOOCV_test);
if lbl_predict== lbl_original
accuracy (i)=1
else
    accuracy (i)=0
end
end
mean(accuracy)