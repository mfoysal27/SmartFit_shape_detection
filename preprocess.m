clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\simulated_database\hourglass\';
%%
i=1;
for i=1:numel(folder)
    I=imread(strcat(folder, num2str(i), '.jpg'));
    I=imresize(I, [600 300]);
    I=rgb2gray(I);
%     imshow(I);
    I2=edge(I, 'sobel');
    imshow(I2);
%     se =  strel('line',3,3);
%     I3=imgaussfilt(imdilate(mat2gray(I2), se), 1);
%         imshow(I3)

    Filename=sprintf('%d_.png',i);
imwrite(I2, Filename);
if i==21
    break
end
end
