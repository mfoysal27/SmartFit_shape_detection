clear all
close all
addpath(genpath(pwd))

I=imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\imdatabase\hourglass\15_.png');

I= 255 * repmat(uint8(I), 1, 1, 3);
I2=imgaussfilt(I, 2);
subplot(1, 2, 1)
imshow(I);
title ('Original Image');
subplot(1, 2, 2)
imshow(I2);
title ('Smoothed Image');
