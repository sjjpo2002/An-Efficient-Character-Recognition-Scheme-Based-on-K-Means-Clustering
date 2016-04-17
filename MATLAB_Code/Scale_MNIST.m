clear all;
clc;
close all;
%% Training Data Preparation
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
scaled_image1=imresize(original_image,0.8,'nearest');
scaled_image2=imresize(original_image,0.8,'bilinear','Antialiasing',false);
scaled_image3=imresize(original_image,0.8,'bicubic','Antialiasing',false);
subplot(3,4,1);
    imshow(original_image), title('Original Image');
subplot(3,4,2);
    imshow(scaled_image1), title('Nearest Neighbor');
subplot(3,4,3);
    imshow(scaled_image2), title('Bilinear');
subplot(3,4,4);
    imshow(scaled_image3), title('Bicubic');
original_image=reshape(uint8(trnd{5}(:,3879)),28,28)';
scaled_image1=imresize(original_image,0.8,'nearest');
scaled_image2=imresize(original_image,0.8,'bilinear','Antialiasing',false);
scaled_image3=imresize(original_image,0.8,'bicubic','Antialiasing',false);
subplot(3,4,5);
    imshow(original_image);
subplot(3,4,6);
    imshow(scaled_image1);
subplot(3,4,7);
    imshow(scaled_image2);
subplot(3,4,8);
    imshow(scaled_image3);
original_image=reshape(uint8(trnd{9}(:,3551)),28,28)';
scaled_image1=imresize(original_image,2,'nearest');
scaled_image2=imresize(original_image,2,'bilinear');
scaled_image3=imresize(original_image,2,'bicubic');
subplot(3,4,9);
    imshow(original_image);
subplot(3,4,10);
    imshow(scaled_image1);
subplot(3,4,11);
    imshow(scaled_image2);
subplot(3,4,12);
    imshow(scaled_image3);