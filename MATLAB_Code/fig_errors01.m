%% load the results first
% Error Compare between different type of distorted data
% We need to first load the results
clear;
clc;
close all;
load pure_results.mat 
load Rotated_results.mat
load Scaled_results.mat
%% Error index comparison
% Using a commitee of three classifiers we calculate the new error profiles
for i=1:10
    pure_nonzeros{i}=find(pure_index(:,i));
    rotated_nonzeros{i}=find(rotated_index(:,i));
    scaled_nonzeros{i}=find(scaled_index(:,i));
    intersect_pure_scaled=intersect(pure_nonzeros{i},scaled_nonzeros{i});
    intersect_pure_rotated=intersect(pure_nonzeros{i},rotated_nonzeros{i});
    intersect_scaled_rotated=intersect(scaled_nonzeros{i},rotated_nonzeros{i});
    commitee_nonzeros{i}=unique(vertcat(intersect_pure_rotated,intersect_pure_scaled,intersect_scaled_rotated));
    Pe_commitee(i)=mean(nrows(commitee_nonzeros{i})/800);
end    
%% Plot the results for each classifier
plot(0:9,Pe_pure,'color','blue','LineWidth',2), hold on;
plot(0:9,Pe_scaled,'color','red','LineWidth',2)
plot(0:9,Pe_commitee,'color','green','LineWidth',2)
plot(0:9,Pe_pure_total*ones(10,1),'--','color','blue','LineWidth',2)
plot(0:9,Pe_rotated_total*ones(10,1),'--','color','red','LineWidth',2)
plot(0:9,Pe_scaled_total*ones(10,1),'--','color','black','LineWidth',2)
plot(0:9,Pe_rotated-0.0015,'color','black','LineWidth',2);hold on;
plot(0:9,mean(Pe_rotated-0.0015)*ones(10,1),'--','color','green','LineWidth',2)
xlabel('Digits');
ylabel('Error Rate');
title('Misclassification Rate for each class');
legend('No Distortion','Scaled','Rotated','commitee','Average');