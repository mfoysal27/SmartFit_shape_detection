addpath(genpath(pwd));


imds = imageDatastore('simulated_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');
% imds = imageDatastore('smartfit_database', 'IncludeSubfolders', true,  'LabelSource', 'foldernames');


[trainingSet, validationSet] = splitEachLabel(imds, .7, 'randomize');

bag = bagOfFeatures(trainingSet, 'StrongestFeatures', 0.5);
opts = templateSVM('BoxConstraint',0.75,'KernelFunction','gaussian');
classifier = trainImageCategoryClassifier(trainingSet,bag,'LearnerOptions',opts);

predict(classifier, validationSet);
confMatrix = evaluate(classifier, validationSet)
accuracy=mean(diag(confMatrix))
