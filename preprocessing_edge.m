clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\';
%%
    I=imread(strcat(folder, 'simulated_image_1', '.jpg'));
    I2=rgb2gray(I);
    I3=uint8(edge(I2));
    I3_smooth=imgaussfilt(I3, 0.5);
    imshow(I3_smooth, []);
    
%     avgH = integralKernel([1 1 7 7], 1/49);
%    J = integralFilter(I3, avgH);
%    J = uint8(J);
%    figure
%    imshow(J);
    
    
    
    %%
        boxfilter=rgb2gray(imread(strcat(folder, '9x9', '.jpg')));
I4=conv2 (I3, boxfilter);
imshow(I4, [])

% %%
% 
points = detectSURFFeatures(I2);
imshow(I4, []); hold on;
plot(points.selectStrongest(40));


