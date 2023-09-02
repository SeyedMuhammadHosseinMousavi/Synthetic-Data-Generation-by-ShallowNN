%% Synthetic Data Generation by shallowNN
clear;
% Load the original dataset
load fisheriris.mat;
original_data = meas';
Classes=3; % Number of Classes
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
% Normalize the data to [0, 1] range
min_val = min(original_data, [], 2);
max_val = max(original_data, [], 2);
normalized_data = (original_data - min_val) ./ (max_val - min_val);

% Define autoencoder architecture
input_size = size(normalized_data, 1);
hidden_size = 5; % Size of the compressed representation
output_size = input_size;

autoencoder = feedforwardnet(hidden_size);
% 'trainlm'	    Levenberg-Marquardt
% 'trainbr' 	Bayesian Regularization (good)
% 'trainrp'  	Resilient Backpropagation
% 'traincgf'	Fletcher-Powell Conjugate Gradient
% 'trainoss'	One Step Secant (good)
% 'traingd' 	Gradient Descent
autoencoder.trainFcn = 'trainoss'; % You can choose different training algorithms
autoencoder = train(autoencoder, normalized_data, normalized_data);

% Generate synthetic data using the trained autoencoder

num_samples = 500; % Number of generating samples

synthetic_data_normalized = rand(input_size, num_samples);
synthetic_data_normalized = autoencoder(synthetic_data_normalized);

% Denormalize synthetic data
synthetic_data = synthetic_data_normalized .* (max_val - min_val) + min_val;
synthetic_data_normalized=synthetic_data_normalized';

original_data=original_data';
synthetic_data=synthetic_data';
Syn=synthetic_data;

%% Getting labels of synthetic generated data by K-means clustering
[Lbl,C,sumd,D] = kmeans(Syn,Classes,'MaxIter',10000,...
    'Display','final','Replicates',10);
SynAll= [Syn Lbl];
SynSort = sortrows(SynAll,5);
Syn=SynSort(:,1:end-1);
Lbl=SynSort(:,end);

%% Plot data and classes
Feature1=1;
Feature2=4;
f1=meas(:,Feature1); % feature 1
f2=meas(:,Feature2); % feature 2
ff1=Syn(:,Feature1); % feature 1
ff2=Syn(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1)
area(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,2)
area(Syn, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,3)
gscatter(f1,f2,Target,'rgm','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,4)
gscatter(ff1,ff2,Lbl,'rgm','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,[5 6])
histogram(meas, 'Normalization', 'probability', 'DisplayName', 'Original Data');
hold on;
histogram(Syn, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
legend('Original','Synthetic')

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(Syn,Lbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm);
SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);

%% Test error and accuracy calculations
DataSize=size(meas);DataSize=DataSize(1,1);
a=0;b=0;c=0;
for i=1:DataSize
if label5(i)== 1
a=a+1;
elseif label5(i)==2
b=b+1;
else
label5(i)==3
c=c+1;
end;end;
erra=abs(a-50);errb=abs(b-50);errc=abs(c-50);
err=erra+errb+errc;TestErr=err*100/DataSize;
SVMAccAugTest=100-TestErr; % Test Accuracy

%% Train and Test Accuracy Results
AugRessvm = [' SDG Train SVM "',num2str(SVMAccAugTrain),'" SDG Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugRessvm);

