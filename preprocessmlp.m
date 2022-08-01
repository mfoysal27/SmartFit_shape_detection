clc
clear all
addpath(genpath(pwd))
folder='C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\simulated_database - Copy\Inverted Triangle\';
%%
i=1;
for i=1:numel(folder)
    I=imread(strcat(folder, '1 (', num2str(i), ').jpg'));
    I=imresize(I, [227 227]);
   Filename=sprintf('%d_.jpg',i);
imwrite(I, Filename);
if i==21
    break
end
end
