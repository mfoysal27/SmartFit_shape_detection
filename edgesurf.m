clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\';
%%
    I=imread(strcat(folder, 'simulated_database\hourglass\1 (4)', '.jpg'));
    I=rgb2gray(I);
    imshow(I);
    I2=uint8(edge(I, 'sobel'));
    imshow(I2);
    %
        I3=imgaussfilt(I2, 2);
        filter= rgb2gray(imread(strcat(folder, '9x9', '.jpg')));
    I4= conv2 (I3, filter, 'full');
%     imshow(I4)
    
    %%
    I5=imread('simulation_image.jpg');
points = detectSURFFeatures(I);
imshow(I3, []); hold on;
plot(points.selectStrongest(40));