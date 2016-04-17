clear all;
clc;
close all;
%% Training Data Preparation
load train_data      % trnd will be created in workplace
% trnd is 10*1 a cell array. Each cell element is a 784*6000 matrix. So, every column in this element is a sample. 
% To apply the rotation transformation we need to convert the image from a
% 784*1 array to a 28*28 square image. This can be done using reshape
% command. So, the best method is to reshape all the samples and then apply
% the rotation to all the samples in the dataset
for i=1:10
    for j=1:ncolumns(trnd{i})
        original_image=reshape(uint8(trnd{i}(:,j)),28,28)';
        scaled_image=imresize(original_image,0.9,'bicubic');
        scaled_trnd{i,1}(:,j)=double(reshape(scaled_image',nrows(scaled_image)*ncolumns(scaled_image),1));
    end
end
% select a random image to show
r1=ceil(10*rand);
r2=ceil(5000*rand);
subplot(1,2,1);
     imshow(reshape(uint8(trnd{r1}(:,r2)),28,28)'), title('Original Image');
subplot(1,2,2);
     imshow(reshape(uint8(scaled_trnd{r1}(:,r2)),nrows(scaled_image),ncolumns(scaled_image))'), title('Scaled Image');
 save('scaled_trnd.mat','scaled_trnd');
 %% Test Data Preparation
load test_data      % tstd will be created in workplace
% trnd is 10*1 a cell array. Each cell element is a 784*6000 matrix. So, every column in this element is a sample. 
% To apply the rotation transformation we need to convert the image from a
% 784*1 array to a 28*28 square image. This can be done using reshape
% command. So, the best method is to reshape all the samples and then apply
% the rotation to all the samples in the dataset
for i=1:10
    for j=1:ncolumns(tstd{i})
        original_image=reshape(uint8(tstd{i}(:,j)),28,28)';
        scaled_image=imresize(original_image,0.9,'bicubic');
        scaled_tstd{i,1}(:,j)=double(reshape(scaled_image',nrows(scaled_image)*ncolumns(scaled_image),1));
    end
end
% select a random image to show
r1=ceil(10*rand);
r2=ceil(800*rand);
figure;
subplot(1,2,1);
     imshow(reshape(uint8(tstd{r1}(:,r2)),28,28)'), title('Original Test Image');
subplot(1,2,2);
     imshow(reshape(uint8(scaled_tstd{r1}(:,r2)),nrows(scaled_image),ncolumns(scaled_image))'), title('Scaled Test Image');
 save('scaled_tstd.mat','scaled_tstd');