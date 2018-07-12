function [Results_SVM_RFE] = SVM_RFE(Data, Label, Names)

%    SVM with recursive feature elimination
%    This function was written by Atif Khan @ University of Rochester NY
%    Last Modified: 12-July-2018
%    Example of function usage: Results_SVM_RFE = SVM_RFE(Data,Label,Names)
%    where:
%    Data: the original data matrix  where rows represent subjects and coulmns represent the features 
%    Label: the binary class label for SVM 
%    Names: the name of features or variable names 
%    The results are stored as a strucure 'Results_SVM_RFE' and excel file named 'SVM_RFE_Results.xlsx'

tic  % Start the clock, for performance evualtion. 

% Get user input for standarizing the data before SVM classification 
prompt = '\n Please enter 0 to use raw data otherwise enter 1 to use standarized data (zscores)  \n\n';
x = input(prompt);

% Error handling if the input is not 0 or 1. 
while (x~=0) && (x~=1) 
  % fprintf(['\n [' 8 'Invalid input. Please enter a valid option from below:]' 8 ' \n']);
    fprintf(2,'\n Invalid Input. Please enter a valid option from below:\n')
    x = input(prompt);
end; 

% Set standardization flag 
if x==0;
    std={'false'};    
else x==1;
    std={'true'};   
end; 
    
    
Size_of_Data=size(Data);   % get size of the data
closs=0; misclass=0;    % initialize the variables to store SVM classification performance 

for i = 1:Size_of_Data(1,2)   % Iterate the loop with the total number of features 

size_of_data=size(Data); % Get the dimensions of reduced datasets  
SVMModel = fitcsvm((Data),Label,'KernelFunction','linear','Standardize',true); % Train SVM model with linear kernal
closs(i,:) = resubLoss(SVMModel);  % Get the accuracy of SVM 
CVSVMModel = crossval(SVMModel);   % Cross validate the SVM model 
misclass(i,:) = kfoldLoss(CVSVMModel); % Perform k-fold cross validation 
Bias(i,:)= SVMModel.Bias;   % Get the bias values 
Beta_act(1:size_of_data(1,2),i)=SVMModel.Beta; % Get the weights from SVM Classifier

clear Beta;   % Clear the previously stored weights 
Beta=abs(SVMModel.Beta)/max(abs(SVMModel.Beta));  % Get the normalized weights 
[Beta_Sorted,Beta_Index] = sort(Beta,'descend');  % Store the SVM weights 
Beta_removed(1,i)=Beta_Sorted(end,1);             % Remove the variable with least abs weight
Beta_removed_name(1,i)=Names(Beta_Index(end,1));  % Get the name of feature with lowest weight  

Variable_Name(1:size_of_data(1,2),i) = transpose(Names(Beta_Index(1:end,1),1)); 
Variable_Name_Act(1:size_of_data(1,2),i) = transpose(Names); % Get and store the name of the feature removed in each iteration 

Beta_all(Beta_Index(1:end,1),i)=Beta_Sorted(Beta_Index(1:end,1),1);  % Store SVM feature weights 
Data = Data(:,Beta_Index(1:end-1,1)); % Filter the data by removing least significant feature
Names = Names(Beta_Index(1:end-1,1),1); % Fitler the feature names matirx accordingly  

%% Section for ROC and AUC Calcuation 
mdlSVM = fitPosterior(SVMModel);       % Calculate the posterior probabilities 
[~,score_svm] = resubPredict(SVMModel);
[Xsvm_temp,Ysvm_temp,Tsvm_temp,AUCsvm(i)] = perfcurve(Label,score_svm(:,2),1); % Store ROC and AUC Values for Training Set. 
Xsvm(1:length(Xsvm_temp),i)=Xsvm_temp;
Ysvm(1:length(Ysvm_temp),i)=Ysvm_temp;
Tsvm(1:length(Tsvm_temp),i)=Tsvm_temp;


%% Section for CV partitions ROC-AUC. 
[~,score_CV,cost] = kfoldPredict(CVSVMModel);    % Test Classifier with 10 fold CV
[Xsvm_temp_CV,Ysvm_temp_CV,Tsvm_temp_CV,AUCsvm_CV(i)] = perfcurve(Label,score_CV(:,2),1);  % ROC-AUC for Test Set in each CV Partition
Xsvm_CV(1:length(Xsvm_temp_CV),i)=Xsvm_temp_CV;
Ysvm_CV(1:length(Ysvm_temp_CV),i)=Ysvm_temp_CV;
Tsvm_CV(1:length(Tsvm_temp_CV),i)=Tsvm_temp_CV;
i  % Displays the current iteration number in a loop. 
end;

closs_percent=(1-closs)*100;    % Trainig accuracy percentage 
misclass_percent=(1-misclass)*100; % Cross validation accuracy percentage 
% Writing results to an excel file. 
% warning('off','MATLAB:xlswrite:AddSheet');   
% filename = 'SVM_RFE_Results.xlsx';     
% xlswrite(filename,closs_percent,'Closs');
% xlswrite(filename,misclass_percent,'Misclass');
% xlswrite(filename,Bias,'Bias');
% xlswrite(filename,Beta_all,'Beta_All');
% xlswrite(filename,Variable_Name,'Variable_Name');
% xlswrite(filename,Beta_removed_name,'Attributes_Removed'); 

 % Storing results as a structure for future use. 
Results_SVM_RFE=struct('Training_Accuracy',closs_percent,'Cross_Validation_Accuracy',misclass_percent,'Beta_RFE_Actual',Beta_act, 'Attribute_Names_Actual',{Variable_Name_Act}, 'Beta_RFE_Normalized_Sorted',Beta_all, 'Bias_All', Bias, 'Attribute_Names_Sorted',{Variable_Name},'Attribute_Removed',{Beta_removed_name},'Xsvm',Xsvm,'Ysvm',Ysvm,'Tsvm',Tsvm,'AUCsvm',AUCsvm, 'Xsvm_CV',Xsvm_CV,'Ysvm_CV',Ysvm_CV,'Tsvm_CV',Tsvm_CV,'AUCsvm_CV',AUCsvm_CV);

%Display on successfull run. 
fprintf(['\n [' 8 'SVM-RFE results are successfully loaded to the workspace and also to the Results_SVM_RFE.xlsx file.]' 8 ' \n\n']);
toc   % End the clock and display total elapsed time. 

end



