clc;clear;close all;
examples_folder_path = 'D:/Files/WHsProjects/sEMG_IR/WAK20220824/features_standard/20220824205141';
movement_type = 'WAK';
fig1_savepath = 'D:\Files\WHsProjects\RMSE_WAK_Regression20220824.jpg';
fig2_savepath = 'D:\Files\WHsProjects\WAK_Regression20220824.jpg';



%% load data
examples_head_path = fullfile(examples_folder_path,'Head.mat');
load(examples_head_path)
load(path)

subjects_num = max(size(subjects_list));
[movements_serial, num_labels_serial, OneHot_labels_serial] = get_serial__labels(movements_label_list, num_label_list, OneHot_label_list);

examples = reshape(examples,size(examples,1),size(examples,2)*size(examples,3));
knee_angle_examples = [examples, angle_labels(:,3)];
ankle_angle_examples = [examples, angle_labels(:,4)];
knee_toruqe_examples = [examples, torque_labels(:,3)];
ankle_toruqe_examples = [examples, torque_labels(:,4)];

subjects_results ={};
for subject_check = 1:subjects_num
    subject= subjects_list(subject_check);
    subject_name = ['Subject',num2str(subject)];
    
    knee_angle_examples_pre_subject = knee_angle_examples(details_subject == subject,:);
    [knee_angle_trainedModel, knee_angle_validationRMSE, knee_angle_true_labels, knee_angle_predict_labels] = trainRegressionModel(knee_angle_examples_pre_subject);

    ankle_angle_examples_pre_subject = ankle_angle_examples(details_subject == subject,:);
    [ankle_angle_trainedModel, ankle_angle_validationRMSE, ankle_angle_true_labels, ankle_angle_predict_labels] = trainRegressionModel(ankle_angle_examples_pre_subject);
    
    knee_toruqe_examples_pre_subject = knee_toruqe_examples(details_subject == subject,:);
    [knee_toruqe_trainedModel, knee_toruqe_validationRMSE, knee_toruqe_true_labels, knee_toruqe_predict_labels] = trainRegressionModel(knee_toruqe_examples_pre_subject);
    
    ankle_toruqe_examples_pre_subject = ankle_toruqe_examples(details_subject == subject,:);
    [ankle_toruqe_trainedModel, ankle_toruqe_validationRMSE, ankle_toruqe_true_labels, ankle_toruqe_predict_labels] = trainRegressionModel(ankle_toruqe_examples_pre_subject);
    
    validationRMSE = [knee_angle_validationRMSE, ankle_angle_validationRMSE, knee_toruqe_validationRMSE, ankle_toruqe_validationRMSE];
    true_labels = [knee_angle_true_labels, ankle_angle_true_labels, knee_toruqe_true_labels, ankle_toruqe_true_labels];
    predict_labels = [knee_angle_predict_labels, ankle_angle_predict_labels, knee_toruqe_predict_labels, ankle_toruqe_predict_labels];
    joints = {'Knee angle', 'Ankle angle', 'Knee torque', 'Ankle torque'};
    
    results_output_regression(subject_name, validationRMSE, joints)
    subject_result = {subject_name, validationRMSE, true_labels, predict_labels};
    subjects_results{end+1} = subject_result;
end

%% plot
sub_RMSE_pre_subject = zeros(subjects_num,length(joints));
for subject_check = 1:subjects_num
    validationRMSE = subjects_results{subject_check}{2};
    for joint_check = 1:length(joints)
        sub_RMSE_pre_subject(subject_check, joint_check) = validationRMSE(joint_check);
    end
end

fig1 = figure('color','w');
set(gcf,'position',[50,50,1600,400])
subplot(1,1,1,'position',[0.08,0.15,0.9,0.7])
hold on
grid on
bar_name = categorical(joints, joints,'Ordinal',true);
ave_sub_RMSE_pre_subject = mean(sub_RMSE_pre_subject,1);
bar(bar_name,ave_sub_RMSE_pre_subject,'FaceColor',[0.8,0.8,0.8],'EdgeColor',[0.5,0.5,0.5],'linewidth',0.01,'basevalue',0.001)
for subject_check = 1:subjects_num
    plot(sub_RMSE_pre_subject(subject_check,:),'o-','linewidth',1.5)
end
title(['RMSE of Gaussian Process Regression on ', movement_type],'fontsize',30)
% xlabel('Joints','FontSize',30)
ylabel('RMSE','FontSize',30)
set(gca,'FontSize',25)
saveas(fig1, fig1_savepath)

fig2 = figure('color','w');
hold on
set(gcf,'position',[50,50,1600,500])
subject_check = 10;
subplot(1,1,1,'position',[0.1,0.25,0.85,0.65])
hold on
plot(subjects_results{subject_check}{3}(:,1),'r-','linewidth',1.5)
plot(subjects_results{subject_check}{4}(:,1),'r--','linewidth',1.5)
plot(subjects_results{subject_check}{3}(:,2),'b-','linewidth',1.5)
plot(subjects_results{subject_check}{4}(:,2),'b--','linewidth',1.5)
plot(subjects_results{subject_check}{3}(:,3),'m-','linewidth',1.5)
plot(subjects_results{subject_check}{4}(:,3),'m--','linewidth',1.5)
plot(subjects_results{subject_check}{3}(:,4),'g-','linewidth',1.5)
plot(subjects_results{subject_check}{4}(:,4),'g--','linewidth',1.5)

title(['Gaussian Process Regression Results of Subject', num2str(subjects_list(subject_check))],'fontsize',30)
xlabel('Sample Points','FontSize',30)
ylabel('Angle(бу)/Torque(Nm)','FontSize',30)
set(gca,'FontSize',25,'linewidth',1.5)
set(gca,'xtick',[],'XLim',[1 max(size(subjects_results{subject_check}{3}(:,1)))])

legend('Knee angle true', 'Knee angle predict','Ankle angle true', 'Ankle angle predict', 'Knee torque true','Knee torque predict', 'Ankle torque true', 'Ankle torque predict','NumColumns',4,'FontSize',20)
legend('location',[0,0.025,1,0.1])
legend('boxoff')
saveas(fig2, fig2_savepath)


%% function
function [trainedModel, validationRMSE, response, validationPredictions] = trainRegressionModel(trainingData)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% returns a trained regression model and its RMSE. This code recreates the
% model trained in Regression Learner app. Use the generated code to
% automate training the same model with new data, or to learn how to
% programmatically train models.
%
%  Input:
%      trainingData: a matrix with the same number of columns and data type
%       as imported into the app.
%
%  Output:
%      trainedModel: a struct containing the trained regression model. The
%       struct contains various fields with information about the trained
%       model.
%
%      trainedModel.predictFcn: a function to make predictions on new data.
%
%      validationRMSE: a double containing the RMSE. In the app, the
%       History list displays the RMSE for each model.
%
% Use the code to train the model with new data. To retrain your model,
% call the function from the command line with your original data or new
% data as the input argument trainingData.
%
% For example, to retrain a regression model trained with the original data
% set T, enter:
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% To make predictions with the returned 'trainedModel' on new data T2, use
%   yfit = trainedModel.predictFcn(T2)
%
% T2 must be a matrix containing only the predictor columns used for
% training. For details, enter:
%   trainedModel.HowToPredict

% Auto-generated by MATLAB on 2022-08-24 22:18:52

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_55;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
regressionGP = fitrgp(...
    predictors, ...
    response, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'rationalquadratic', ...
    'Standardize', false);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
gpPredictFcn = @(x) predict(regressionGP, x);
trainedModel.predictFcn = @(x) gpPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RegressionGP = regressionGP;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2018b.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 54 columns because this model was trained using 54 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_55;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionGP, 'KFold', 5);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
end

function [movements, num_labels, OneHot_labels] = get_serial__labels(movements_label_list, num_label_list, OneHot_label_list)
movements = {};
num_labels = {};
OneHot_labels = {};
for label_check = 1:max(size(num_label_list))
    min_sub = find(num_label_list == min(num_label_list));
%     moviement_name = strtrim(string(movements_label_list(min_sub,:)));
    moviement_name = strtrim(movements_label_list(min_sub,:));
    if strcmp(moviement_name,'R') ~= 1
        movements = [movements, moviement_name];
        num_labels = [num_labels, num2str(num_label_list(min_sub))];
        OneHot_labels = [OneHot_labels, num2str(OneHot_label_list(min_sub,:))];
    end
    num_label_list(min_sub) = inf;
end
end

function results_output_regression(subject_name, validationRMSE,joints)
Sub_RMSE = [];
for joint_check = 1:max(size(joints))
    Sub_RMSE = [Sub_RMSE, joints{joint_check}, ':', num2str(validationRMSE(joint_check)),'   '];
end
fprintf([subject_name,'   :\n'])
fprintf([Sub_RMSE,'\n','\n'])
end
