clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\smartfit_database_preprocessed\';
folder1='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\im_smartfit_CNN_database\';

Alexnet_size=[227 227];

%%
for i=1:20
    I=imread(strcat(folder,'hourglass\',  num2str(i), '_.png'));
    I=imresize(I, Alexnet_size);
    I2=uint8( I(:,:,[1 1 1]) * 255 );
Filename=(strcat(folder1,'hourglass\',  num2str(i), '.png'));
imwrite(I2, Filename);
end

for i=1:20
    I=imread(strcat(folder,'inverted triangle\',  num2str(i), '_.png'));
    I=imresize(I,Alexnet_size);
    I2=uint8( I(:,:,[1 1 1]) * 255 );
Filename=(strcat(folder1,'inverted triangle\',  num2str(i), '.png'));
imwrite(I2, Filename);
end

for i=1:20
    I=imread(strcat(folder,'pear\',  num2str(i), '_.png'));
    I=imresize(I, Alexnet_size);
    I2=uint8( I(:,:,[1 1 1]) * 255 );
Filename=(strcat(folder1,'pear\',  num2str(i), '.png'));
imwrite(I2, Filename);
end

for i=1:20
    I=imread(strcat(folder,'rectangle\',  num2str(i), '_.png'));
    I=imresize(I, Alexnet_size);
    I2=uint8( I(:,:,[1 1 1]) * 255 );
Filename=(strcat(folder1,'rectangle\',  num2str(i), '.png'));
imwrite(I2, Filename);
end