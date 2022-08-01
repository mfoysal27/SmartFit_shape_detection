clear all
close all
addpath(genpath(pwd))

I=mat2gray(imread('C:\Users\kamru\OneDrive - Texas Tech University\Desktop\Fall 2020\Smartfit Algorithm\imdatabase\hourglass\15_.png'));
Filter=rgb2gray(imread('9x9.jpg'));
I_final=conv2 (I, Filter, 'full');
imshow(I_final);
I_suppressed=nonmaxsup2d(I_final);

function imgResult = nonmaxsup2d(imgHough)
imgResult = zeros(size(imgHough));
for y = 2:size(imgHough, 1)-1
    for x = 2:size(imgHough, 2)-1
        offx = [1 1 0 -1 -1 -1  0  1];
        offy = [0 1 1  1  0 -1 -1 -1];
        val = imgHough(y, x);
        is_max = true;
        for i=1:8
            if y == 2 && offy(i) == -1
                continue
            end
            if y ==size(imgHough,1)-1 && offy(i) == 1
                continue
            end
            if x ==2 && offx(i) == -1
                continue
            end
            if x ==size(imgHough,2)-1 && offx(i) == 1
                continue
            end
            if val < imgHough(y+offy(i), x+offx(i))
                is_max = false;
                break;
            end
        end
        if is_max
            imgResult(y, x) = val;
        end
    end
end
end