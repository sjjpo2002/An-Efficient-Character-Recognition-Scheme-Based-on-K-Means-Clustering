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
Deg=[10 10 10 10 10 10 10 10 10 10];
% Deg=[10 300 10 10 10 10 30 110 90 40];   Optimum
for i=1:10
    if Deg(i)~=0        
        for j=1:ncolumns(trnd{i})
             original_image=reshape(uint8(trnd{i}(:,j)),28,28)';
             rotated_image=imrotate(original_image,Deg(i),'bilinear','crop');
             rotated_trnd{i,1}(:,j)=double(reshape(rotated_image',784,1));
        end
    else
        rotated_trnd(i,1)=trnd(i); 
    end
end
% select a random image to show
r1=ceil(10*rand);
r2=ceil(5000*rand);
subplot(1,2,1);
     imshow(reshape(uint8(trnd{r1}(:,r2)),28,28)'), title('Original Image');
subplot(1,2,2);
     imshow(reshape(uint8(rotated_trnd{r1}(:,r2)),28,28)'), title('Rotated Image');
 save('rotated_trnd.mat','rotated_trnd');
 % ============================ Prepare Test Data =========
 load test_data      % tstd will be created in workplace
% tstd is 10*1 a cell array. Each cell element is a 784*1000 matrix. So, every column in this element is a sample. 
% To apply the rotation transformation we need to convert the image from a
% 784*1 array to a 28*28 square image. This can be done using reshape
% command. So, the best method is to reshape all the samples and then apply
% the rotation to all the samples in the dataset
for i=1:10
    if Deg(i)~=0  
        for j=1:ncolumns(tstd{i})
            original_image=reshape(uint8(tstd{i}(:,j)),28,28)';
            rotated_image=imrotate(original_image,Deg(i),'bilinear','crop');
            rotated_tstd{i,1}(:,j)=double(reshape(rotated_image',784,1));
        end
    else
        rotated_tstd(i,1)=tstd(i); 
    end
end
% select a random image to show
r1=ceil(10*rand);
r2=ceil(800*rand);
figure;
subplot(1,2,1);
     imshow(reshape(uint8(tstd{r1}(:,r2)),28,28)'), title('Original Test Image');
subplot(1,2,2);
     imshow(reshape(uint8(rotated_tstd{r1}(:,r2)),28,28)'), title('Rotated Test Image');
 save('rotated_tstd.mat','rotated_tstd');