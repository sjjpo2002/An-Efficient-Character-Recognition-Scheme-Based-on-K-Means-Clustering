clear all;
clc;
close all;
% Training Data Preparation
load train_data      % trnd will be created in workplace
c = max(size(trnd));  % No. of classes which is 10 for the digit input data
NpC = 5000;           % No. of samples per digits from trainig data, <5400 recommended   
TstNpC = 800;         % No. of samples per digits from test data, <890 recommended 
NT = c*NpC;           % total No. of training samples
SpSpec = 3*ones(1,c); % splitting spec, i-th elements is number of clusters within i-th class
% SpSpec = [2 3 3 4 3 2 3 4 4 4]; % optimum spec
% SpSpec = ones(1:c)  % no k-means clustering for ordinary LDA 
Spc = sum(SpSpec);    % No. of split classes afetr clustering
original_image=reshape(uint8(trnd{7}(:,187)),28,28)';
rotated_image1=imrotate(original_image,20,'nearest','crop');
rotated_image2=imrotate(original_image,20,'bilinear','crop');
rotated_image3=imrotate(original_image,20,'bicubic','crop');
subplot(3,4,1);
    imshow(original_image), title('Original Image');
subplot(3,4,2);
    imshow(rotated_image1), title('Nearest Neighbor');
subplot(3,4,3);
    imshow(rotated_image2), title('Bilinear');
subplot(3,4,4);
    imshow(rotated_image3), title('Bicubic');
original_image=reshape(uint8(trnd{2}(:,3879)),28,28)';
rotated_image1=imrotate(original_image,20,'nearest','crop');
rotated_image2=imrotate(original_image,20,'bilinear','crop');
rotated_image3=imrotate(original_image,20,'bicubic','crop');
subplot(3,4,5);
    imshow(original_image);
subplot(3,4,6);
    imshow(rotated_image1);
subplot(3,4,7);
    imshow(rotated_image2);
subplot(3,4,8);
    imshow(rotated_image3);
original_image=reshape(uint8(trnd{4}(:,3331)),28,28)';
rotated_image1=imrotate(original_image,20,'nearest','crop');
rotated_image2=imrotate(original_image,20,'bilinear','crop');
rotated_image3=imrotate(original_image,20,'bicubic','crop');
subplot(3,4,9);
    imshow(original_image);
subplot(3,4,10);
    imshow(rotated_image1);
subplot(3,4,11);
    imshow(rotated_image2);
subplot(3,4,12);
    imshow(rotated_image3);
