clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\';
%%
    I=imread(strcat(folder, 'simulation_image', '.jpg'));
    I2=rgb2gray(I);
    I3=(integralImage(I2));
    I3=edge(I3);
    imshow(I3, []);
    
    avgH = integralKernel([1 1 7 7], 1/49);
   J = integralFilter(I3, avgH);
   J = uint8(J);
   figure
   imshow(J);
    
    
    
    %%
        boxfilter=rgb2gray(imread(strcat(folder, '15x15', '.jpg')));
I4=conv2 (I3, boxfilter, 'Full');
imshow(I4, [])
I5=edge(I4, 'sobel');
imshow(I5, []);