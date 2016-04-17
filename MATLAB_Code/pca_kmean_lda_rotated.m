clear;
clc;
%% Training Data Preparation
load rotated_trnd.mat; 
digits(128);
c = max(size(rotated_trnd));  % No. of classes which is 10 for the digit input data
NpC = 5000;           % No. of samples per digits from trainig data, <5400 recommeded   
TstNpC = 800;         % No. of samples per digits from test data, <890 recommended 
NT = c*NpC;           % total No. of training samples
% SpSpec = 3*ones(1,c); % splitting spec, i-th elements is number of clusters within i-th class
SpSpec=[2 3 3 4 3 3 3 3 5 4];
% SpSpec=[2 3 3 4 3 3 3 3 5 4]   Optimum
Spc = sum(SpSpec);    % No. of split classes afetr clustering
pca_fac = .9;         % ratio of ignored dimentions to total dimention in PCA <1   
TSum = 0;             % total sum of training samples 
for i=1:c  
    rotated_trnd{i} = rotated_trnd{i}(:,1:NpC);    
    TSum = TSum + sum(rotated_trnd{i},2); % total sum of training samples
end 
Trs = 20;   % threshold to detect dead pixels varying in [0,255], 20 to 40 is recommended
dead_pixels = (TSum<Trs*NT);      % dead pixels index which are off most of times
live_pixels = (TSum>=Trs*NT);     % live pixels index which are used in the follwoing     
dim = sum(live_pixels);           % new dimention is No. of live pixels   
TMean = TSum(live_pixels,:)/NT;   % total mean of training samples
         
%% Dim. Reduction with PCA 
S_t = zeros(dim,dim);              % total scattering
pcadim = floor((1-pca_fac)*dim);   % samples dimentionality following PCA performed
if pcadim < Spc-1
    display(['Too low dimentionality to operate LDA, minimum pca_fac = ',num2str(Spc/dim)])
    TLDA = eye(pcadim);
    Spc = pcadim+1;
end
for i=1:c  
    rotated_trnd{i}  = rotated_trnd{i}(live_pixels,1:NpC); % excluding ever-off pixels to reduce dimentionality
    for j=1:NpC
        S_t = S_t + (rotated_trnd{i}(:,j)-TMean)*(rotated_trnd{i}(:,j)-TMean)'/NT; 
    end 
    clc % counter display      
end
[Upca,Lmbd,~] = svd(S_t); 
Lmbd = diag(Lmbd);
Lmbd = diag(Lmbd(1:pcadim)); % pcadim largest eigen values are selected
TPCA(:,:) = Upca(:,1:pcadim)*sqrt(inv(Lmbd)); 
for i=1:c    
    pcarotated_trnd{i} = TPCA'*rotated_trnd{i};   
end 
pcaS_t = TPCA'*S_t*TPCA; % new class covarince matrix (pcadim)-by-(pcadim)
%% K-Means Clustering over Training Classes 
%k-means is preformmed on each class with k specified in SpSpec. Initial
%means are uniformly selected fronm the samples within each class.
sprotated_trnd = cell(Spc,1); % split data ensemble containing the entire clusters from k-means
for i=1:c 
    [INDX,M] = kmeans((pcarotated_trnd{i}(:,1:NpC))',SpSpec(i));,...
%                       'start',pcarotated_trnd{i}(:,SpSpec(i):floor(NpC/SpSpec(i)):NpC)');
    for j=0:SpSpec(i)-1
        sprotated_trnd{sum(SpSpec(1:i))-j}   = pcarotated_trnd{i}(:,INDX==j+1);         % new split classes
        SCMean(:,sum(SpSpec(1:i))-j) = M(j+1,:)';                       % split class mean
        [~,SNpC(sum(SpSpec(1:i))-j)]= size(sprotated_trnd{sum(SpSpec(1:i))-j}); % No. samples per split class
    end
    clc % counter display    
end  

%% LDA on New Split Classes
% the following LDA transformation reduces the dimentionality to Spc-1
if pcadim >= Spc-1
    S_B = zeros(pcadim,pcadim);       % between class scattering
    for i=1:Spc  
        S_B = S_B + SNpC(i)*(SCMean(:,i)-TPCA'*TMean)*(SCMean(:,i)-TPCA'*TMean)'/NT;
    end
    S_W = pcaS_t - S_B;         % within class scattering having total scattering calculated 
    [Ulda,Lm,~] = svd(S_W\S_B);     % there are Spc-1 non-zero eigen values within Lmbd 
    TLDA = Ulda(:,1:Spc-1);        % LDA trnsform matrix consisting of first Spc-1 eigen vectors
end

%% Transformed Data Spec
load rotated_tstd.mat; 
for m=1:c                    % test ensemble is PCA and LDA transformed 
    newrotated_tstd{m} = TLDA'*TPCA'*rotated_tstd{m}(live_pixels,1:TstNpC); 
 end
for i=1:Spc  
    Cov(:,:,i) = zeros(Spc-1,Spc-1);
    ldarotated_trnd{i} = TLDA'*sprotated_trnd{i}; % clustered training ensemble is LDA transformed  
    Mu(:,i) = TLDA'*SCMean(:,i);              % new class means (Spc-1)-by-Spc
    for j=1:SNpC(i)             % new class covarince matrix (Spc-1)by(Spc-1)by(Spc)
        Cov(:,:,i) = Cov(:,:,i) + (ldarotated_trnd{i}(:,j)-Mu(:,i))*(ldarotated_trnd{i}(:,j)-Mu(:,i))'/SNpC(i);
    end
    dist_term(i) = log(det(Cov(:,:,i)))/2-log(SNpC(i)/NT); % additive term in Mahalanobis Distance
    clc % counter display    
end
%% Test and Error Computation (Mahanalobis)
rotated_index=zeros(TstNpC,10);     % will be one whenever a misprediction occures
for i=1:c
    syndrm = 0;      % error event syndrome
    for j=1:TstNpC
        for k=1:Spc  % squared Mahalanobis distance for each tests sample within each class
            mhd2(k) = (newrotated_tstd{i}(:,j)-Mu(:,k))'*((Cov(:,:,k)\(newrotated_tstd{i}(:,j)-Mu(:,k))))...
                       /2+dist_term(k);          
        end
        [~,I]=min(mhd2);      % minimum ditance index varying from 1 to Spc
        if ((I>sum(SpSpec(1:i)))||(I<=sum(SpSpec(1:i))-SpSpec(i))) 
           syndrm = syndrm+1; % enters when allocated class is not within those for the same digit
           rotated_index(j,i)=1;
        end
    end
    Pe_rotated(i) = syndrm/TstNpC; % misclassification rate for each class 
    clc % counter display
    display(['Percentage Completed: ',num2str(floor(100*i/c)),'%']);
end
Pe_rotated_total = mean(Pe_rotated);  % Average misclassification error
save('Rotated_results','Pe_rotated','Pe_rotated_total','rotated_index')  
display(['Average Error Rate: ',num2str(Pe_rotated_total*100),' %']);